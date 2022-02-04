//! Deep reinforcment learning algorithm for Rattus to predict the location of the next click and with what probability.
//! uses (i32,i32,f32) where the first i32 tuple is the location in x and y coordinates and f32 is the confidence probability of the click from 0-1.

//TODO: allow this to fallback to TF-CPU for machines without GPU and possibly just do inference with a build script.
//TODO: floats stink norm net is an attempt to realizing rot net and uint8 operations with bounded functions while retaining all the flaws in DNNs..

//allow unstable features
// #![feature(int_log)]
use half::bf16;
use half::f16;
use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::result::Result;
//TODO: clean this up with proper heirachy
use tensorflow::ops;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::GradientDescentOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::train::Optimizer;
use tensorflow::Code;
use tensorflow::DataType;
use tensorflow::Graph;
use tensorflow::Operation;
use tensorflow::Output;
use tensorflow::OutputName;
use tensorflow::SavedModelBundle;
use tensorflow::SavedModelSaver;
use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Shape;
use tensorflow::SignatureDef;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::TensorInfo;
use tensorflow::Variable;
use tensorflow::REGRESS_INPUTS;
use tensorflow::REGRESS_METHOD_NAME;
use tensorflow::REGRESS_OUTPUTS;
//import rand
use rand::Rng;
use tensorflow::BFloat16;

/// A standard fully connected layer with bias term
///
/// `activation` is a function which takes a tensor and applies an activation
/// function such as sigmoid.
///
/// Returns variables created and the layer output.
///
fn layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Half)
                .build(w_shape, scope)?,
        )
        .data_type(DataType::Half)
        .shape([input_size, output_size])
        .build(&mut scope.with_op_name("w"))?;
    let b = Variable::builder()
        .const_initial_value(Tensor::<f16>::new(&[output_size]))
        .build(&mut scope.with_op_name("b"))?;
    //n is input_size to be divided at each node in order to normalize the signals at each node before activation
    Ok((
        vec![w.clone(), b.clone()],
        activation(
            ops::add(
                ops::mat_mul(input, w.output().clone(), scope)?,
                b.output().clone(),
                scope,
            )?
            .into(),
            scope,
        )?,
    ))
}

//TODO: extract into class
//================
//----NORM_NET----
//================
/// A standard fully connected layer without bias trainnable parameters
/// instead normalizing and dropping out connections at each node.
///
///
/// #
/// PROS:
///
/// * better exploration by removing instabilities inherent to bias
///        
/// * gradient based connection-wise dropout with tan weights
///       
/// * better transfer learning by removing bias connections
///       
/// * ~shouldn't~ have exploding gradient although vanishing gradient may be possible (also shouldn't be a problem but less hypothesized)
///          due to normalizing division.
///      
/// * technically connection wise dropout is divisive (top down) architecture search (e.g.: the opposite of NEAT (bottom up) which is agglomerative)
///
/// CONS:
/// * may be slower due to more operations
///       
/// * input and output must/should be tailored for normalized input/output (standard data science practices)
///       
/// * needs large type precision for stability, but the stability can be tuned (as apposed to bias which needs architectural considerations)
///
/// #
/// NOTE: parameters goes to zero whereas biases find some 1-dimensional partition from -inf to inf. This helps
/// builds subgraph search modules (subtrees essentially). That can quickly optimize for distinct domains and labels via dropout.
///
/// NOTE:
/// Tanh should be on the first and last layers to map inputs and outputs to negative values.
/// Dividing the input to be between -1 and 1 and multiplying the output by some multiple of
/// 10 allows the otherwise normalized network to take in and output whole integers.
///
/// NOTE:
/// We dont use BFloat since the integer range is only used as a buffer for addition overflow in matmul.
/// In all other operations we are strictly bounded -1 > x > 1. As long as layer_width is not
/// greater than Float range we are fine in the worst case (summing all 1's).
/// Otherwise decimal precision of float type is our parameter type precision.
pub fn norm_layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(w_shape.clone(), scope)?,
        )
        .data_type(DataType::Float)
        .shape([input_size, output_size])
        .build(&mut scope.with_op_name("w"))?;

    let n = ops::constant(input_size as f32, scope)?;

    //NOTE: tan on weights is to force weights to dropout but use the gradient for better dropout than random node based dropout
    //NOTE: we multiply the activation by 100 to represent values </>than 1/-1
    let res = activation(
        //this normalizes to speed up trainning and sample efficiency
        ops::div(
            ops::mat_mul(
                input,
                //this sets the gradients to dropout weights
                ops::tan(w.output().clone(), scope)?,
                scope,
            )?,
            n,
            scope,
        )?
        .into(),
        scope,
    )?
    .into(); //,
    Ok((vec![w.clone()], res))
}

///a normal layer as above but with residual connections
fn norm_layer_res<O1: Into<Output>>(
    input: O1,
    res_input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Half)
                .build(w_shape.clone(), scope)?,
        )
        .data_type(DataType::Half)
        .shape([input_size, output_size])
        .build(&mut scope.with_op_name("w"))?;

    //n is input_size to be divided at each node in order to normalize the signals at each node before activation
    let input_size = 2 * input_size;
    //TODO: concat the input and res tensors
    // let concat = ops::concat(0,vec![input, res_input],  scope)?.into();

    let scalar_coe = ops::constant(f16::from_f32(0.1), scope)?;
    let cur = ops::mat_mul(
        input,
        //NOTE: division for half stability the higher this value the more stable the trainning but the less expressivity (domain) of the weights
        //NOTE: tan on weights is to force weights to dropout but use the gradient for better dropout than random node based dropout
        ops::multiply(
            ops::tan(w.output().clone(), scope)?,
            scalar_coe.clone(),
            scope,
        )?,
        scope,
    )?;

    // let cur_res = ops::mat_mul(
    // res_input,
    // ops::multiply(ops::tan(w_res.output().clone(), scope)?, scalar_coe.clone(), scope)?,
    // scope,
    // )?;
    // ops::tan(w_res.output().clone(), scope)?, scope)?;
    let res_input = ops::multiply(scalar_coe, res_input, scope)?;
    let cur = ops::add(cur, res_input, scope)?;

    let n = ops::constant(f16::from_f32(input_size as f32), scope)?;

    let res = activation(ops::div(cur, n, scope)?.into(), scope)?.into(); //,

    // Ok((vec![w.clone(), w_res.clone()], res))
    Ok((vec![w.clone()], res))
}

/// Creates a fully connected network with normalizing layers.
/// handles all type input and output in the graph, just pass in and expect floats
/// best data wrangling practices are still recommended, especially normalizing each
/// input between -1 > x > 1.
///
///# PARAMETERS:
///
/// * input_size: size of the input vector
///
/// * output_size: size of the output vector
///
/// * layer_width: number of nodes in each layer
///
/// * layer_height: number of layers in the network including input, hidden and output.
///
/// * max_integer: maximum integer value that can be represented by the output of the network
///
/// # RETURNS:
/// * output vector from the network as a tensorflow-rs Output type
///
/// * vector of variables that are the weights of the network as tensorflow-rs Variable type
///
/// * input vector to the network as a tensorflow-rs Operation type
///
/// * output of the TF graph as a tensorflow-rs Operation type
///
/// * the passed in tensorflow-rs mutable scope with the network added as a TF graph
pub fn norm_net(
    scope: &mut Scope,
    input_size: u64,
    output_size: u64,
    layer_width: u64,
    layer_height: u64,
    max_integer: u32,
) -> Result<(Vec<Output>, Vec<Variable>, Operation, Operation, &mut Scope), Status> {
    //TODO: pass in optimizer or just hyperparams?
    //TODO: this may be better served as a builder; or factory for a keras like set of layers
    let input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64, input_size])
        .build(&mut scope.with_op_name("input"))?;
    let label = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape([1u64, output_size])
        .build(&mut scope.with_op_name("label"))?;

    let mut net_vars = vec![];
    let mut net_layers = vec![];

    //initial layer
    let (vars, layer) = norm_layer(
        input.clone(),
        input_size,
        layer_width,
        &|x, scope| Ok(ops::tanh(x, scope)?.into()),
        scope,
    )?;
    net_vars.extend(vars);
    net_layers.push(layer.clone());

    let mut prev_layer = layer;
    //hidden layers
    for i in 0..layer_height - 2 {
        let (vars, layer) = norm_layer(
            prev_layer.clone(),
            layer_width,
            layer_width,
            //NOTE: originally designed with tan but vanishing gradient can occur
            &|x, scope| Ok(ops::tanh(x, scope)?.into()),
            scope,
        )?;
        prev_layer = layer.clone();

        net_vars.extend(vars);
        net_layers.push(layer.clone());
    }

    //the final output layer is tanh to express negative values and multiplied to stabilize the
    //half precision gradient as well as express whole integers outside of -1 and 1.
    let (vars, output) = norm_layer(
        net_layers.last().unwrap().clone(),
        layer_width,
        output_size,
        &|x, scope| {
            Ok(ops::multiply(
                ops::tanh(x, scope)?,
                //TODO: extract this scalar coefficient
                ops::constant(max_integer as f32, scope)?,
                scope,
            )?
            .into())
        },
        scope,
    )?;
    net_vars.extend(vars);
    net_layers.push(output);

    Ok((net_layers, net_vars, input, label, scope))
}

//TODO: save and load from disk
//TODO: does save/load require UUID? allow user to name in initialization for organization?
//TODO: defaults such as learning rate
/// a Normalizing Network currently being researched
/// NOTE: currently inputs and outputs must be flattened if representing >1 dim data
struct NormNet<'a> {
    ///he scope for tensorflow to prevent having multiple scopes active
    scope: &'a mut Scope,
    ///Session options currently being used
    session: Option<Session>,
    session_options: SessionOptions,
    graph: Option<Graph>,
    ///each layers output hook of the network
    net_layers: Vec<Output>,
    ///all the parameters of the network
    net_vars: Vec<Variable>,
    //TODO: these may need to be refcell as well for saved model loader
    Input: Operation,
    Label: Operation,
    Output: Output,
    Error: Operation,
    optimizer: AdadeltaOptimizer,
    minimize_vars: Vec<Variable>,
    minimize: Operation,
    SavedModelSaver: RefCell<Option<SavedModelSaver>>,
}
impl<'a> NormNet<'a> {
    pub fn new(
        scope: &'a mut Scope,
        input_size: u64,
        output_size: u64,
        layer_width: u64,
        layer_height: u64,
        max_integer: u32,
        learning_rate: f32,
        //a power to raise the pythagorean distance error to for gradient scaling error_
        error_power: f32,
    ) -> Result<NormNet, Status> {
        // TODO: consider inlining this since were encapsulating
        let Input = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, input_size])
            .build(&mut scope.with_op_name("input"))?;
        let Label = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape([1u64, output_size])
            .build(&mut scope.with_op_name("label"))?;

        //CONSTRUCT NETWORK
        let mut net_vars = vec![];
        let mut net_layers = vec![];

        //initial layer
        let (vars, layer) = norm_layer(
            Input.clone(),
            input_size,
            layer_width,
            &|x, scope| Ok(ops::tanh(x, scope)?.into()),
            scope,
        )?;
        net_vars.extend(vars);
        net_layers.push(layer.clone());

        let mut prev_layer = layer;
        //hidden layers
        for i in 0..layer_height - 2 {
            let (vars, layer) = norm_layer(
                prev_layer.clone(),
                layer_width,
                layer_width,
                //NOTE: originally designed with tan but vanishing gradient can occur
                &|x, scope| Ok(ops::tanh(x, scope)?.into()),
                scope,
            )?;
            prev_layer = layer.clone();

            net_vars.extend(vars);
            net_layers.push(layer.clone());
        }

        //the final output layer is tanh to express negative values and multiplied to stabilize the
        //half precision gradient as well as express whole integers outside of -1 and 1.
        let (vars, output) = norm_layer(
            net_layers.last().unwrap().clone(),
            layer_width,
            output_size,
            &|x, scope| {
                Ok(ops::multiply(
                    ops::tanh(x, scope)?,
                    ops::constant(max_integer as f32, scope)?,
                    scope,
                )?
                .into())
            },
            scope,
        )?;
        net_vars.extend(vars);
        net_layers.push(output);
        //END OF CONSTRUCTING NETWORK

        let Output = net_layers.last().unwrap().to_owned();

        let options = SessionOptions::new();
        let init_saved_model_saver = RefCell::new(None);

        //TODO: this needs to be in state so that it can be saved for transfer learning
        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(ops::constant(learning_rate, scope)?);

        // DEFINE ERROR FUNCTION //
        //TODO: pass this in conditionally, give user output and label with
        //      a partial constructor then they supply error and construction is complete
        // two structs? whats standard functionally for partial construction?
        //default error is pythagorean distance
        let Error = ops::sqrt(
            ops::pow(
                ops::sub(Output.clone(), Label.clone(), scope)?,
                ops::constant(2.0 as f32, scope)?,
                scope,
            )?,
            scope,
        )?;
        //TODO: at least pass in the constant here as a gradient weighting scalar coefficient
        let Error = ops::pow(
            Error.clone(),
            ops::constant(error_power, scope).unwrap(),
            scope,
        )
        .unwrap();

        let (minimize_vars, minimize) = optimizer
            .minimize(
                scope,
                Error.clone().into(),
                MinimizeOptions::default().with_variables(&net_vars),
            )?
            .into();

        // let mut run_args = SessionRunArgs::new();

        Ok(NormNet {
            scope,
            session: None,
            session_options: options,
            graph: None,
            net_layers,
            net_vars,
            Input,
            Label,
            Output,
            Error,
            optimizer,
            minimize_vars,
            minimize,
            SavedModelSaver: init_saved_model_saver,
        })
    }
    // forward pass and return result
    // fn infer
    //one time traning that can take output of infer and a label
    // fn backprop
    //save the model out to disk in this directory as default
    pub fn save(self) -> Result<(), Box<dyn Error>> {
        // save the model to disk in the current directory
        if self.SavedModelSaver.borrow().is_none() {
            println!("initializing saved model saver..");
            let mut all_vars = self.net_vars.clone();
            all_vars.extend_from_slice(&self.minimize_vars);
            let mut builder = tensorflow::SavedModelBuilder::new();
            builder
                .add_collection("train", &all_vars)
                .add_tag("serve")
                .add_tag("train")
                .add_signature(REGRESS_METHOD_NAME, {
                    let mut def = SignatureDef::new(REGRESS_METHOD_NAME.to_string());
                    def.add_input_info(
                        REGRESS_INPUTS.to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Input.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_output_info(
                        REGRESS_OUTPUTS.to_string(),
                        TensorInfo::new(DataType::Float, Shape::from(None), self.Output.name()?),
                    );
                    //add error
                    // def.add_output_info(
                    //     "error".to_string(),
                    //     TensorInfo::new(DataType::Float, Shape::from(None), Error.name()),
                    // );
                    //add label
                    // def.add_output_info(
                    //     "label".to_string(),
                    //     TensorInfo::new(DataType::Float, Shape::from(None), "Label"),
                    // );
                    def
                });
            let saved_model_saver = builder.inject(self.scope)?;
            self.SavedModelSaver.replace(Some(saved_model_saver));
        }
        // self.SavedModelSaver.borrow_mut().as_mut().unwrap().save();
        // same as above  but pass savedmodelsaver session graph and save directory
        self.SavedModelSaver.borrow_mut().as_mut().unwrap().save(
            &self.session.unwrap(),
            &self.graph.unwrap(),
            "./",
        )?;
        Ok(())
    }

    //TODO: load
    // pub fn load(self) {
    //     // load the model from disk in the current directory
    //     println!("loading previously saved model..");
    //     //TODO: check examples in tensorflow rust for solution:
    //     // load the model from the saved model saver for this training session
    //     let save_dir = std::env::current_dir().unwrap();
    //     let mut graph = Graph::new();
    //     let bundle = SavedModelBundle::load(
    //         //TODO: test here that this overwrites correctly
    //         // &SessionOptions::new(),
    //         &self.session_options,
    //         &["serve", "train"],
    //         &mut graph,
    //         save_dir,
    //     )?;
    //     //TODO: associate the operations with the class state
    //     let session = &bundle.session;
    //     let signature = bundle.meta_graph_def().get_signature(REGRESS_METHOD_NAME)?;
    //     let input_info = signature.get_input(REGRESS_INPUTS)?;
    //     let output_info = signature.get_output(REGRESS_OUTPUTS)?;
    //     let error_info = signature.get_output("error")?;
    //     //TODO: test
    //     // let label_info = signature.get_output("label")?;
    //     //TODO: associate the operations with the class state
    //     //TODO: get variables from collection and associate with the class state, this should be all that is actually needed however.
    //     let input = graph.operation_by_name_required(&input_info.name().name)?;
    //     let output = graph.operation_by_name_required(&output_info.name().name)?;
    //     let error = graph.operation_by_name_required(&self.Error.name()?)?;
    //     // let label = graph.operation_by_name_required(&label_info.name().name)?;
    // }

    /// train the network with the given inputs and labels (must be synchronized in index order)
    /// initializes a saved model saver if not already initialized and saves after trainning.
    ///PARAMETERS:
    /// inputs: the inputs to the network as a flattened 1D vector
    /// labels: the labels for the inputs as a flattened 1D vector, must align index wise with inputs. (e.g. inputs[0] must be labeled as labels[0])
    /// error: the error operation to minimize with (e.g. pythagorean error sqrt(x^2 + y^2))
    /// learning_rate: the learning rate for the network (e.g. 0.001)
    pub fn train<T: tensorflow::TensorType>(
        // &mut self,
        self,
        inputs: Vec<Vec<T>>,
        labels: Vec<Vec<T>>,
        // TODO: pass these is instead of constructing
        // error: Operation,
        // learning_rate: f32,
    ) -> Result<Vec<Tensor<f32>>, Status> {
        //TODO: allow to change learning rate and error here
        // load the model if it has been saved otherwise initialize a saved model saver
        //TODO: extract this so we can call it in other training methods as a "refresh" serialization synchronization
        // TODO: also a sub method for save() and load()

        let g = self.scope.graph_mut();
        let session = Session::new(&self.session_options, &g)?;
        let mut run_args = SessionRunArgs::new();

        // set parameters to be optimization targets
        for var in &self.net_vars {
            run_args.add_target(&var.initializer());
        }
        for var in &self.minimize_vars {
            run_args.add_target(&var.initializer());
        }
        session.run(&mut run_args)?;

        //TODO: shouldnt need to initialize so much ITL
        //TODO: randomize input and labels and k-fold
        //TODO: create dataset structure
        println!("trainning..");
        let mut result = vec![];

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        // print inputs[0].len();
        println!("inputs.len(): {}", inputs.len());
        println!("{}", inputs[0].len());
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);
        println!("{}", labels[0].len());
        // let layer5_output_fetch = run_args.request_fetch(&layer1_output, 0);
        let mut input_iter = inputs.into_iter();
        let mut label_iter = labels.into_iter();
        input_iter.next();
        label_iter.next();
        let mut i = 0;
        loop {
            i += 1;
            let input = input_iter.next();
            let label = label_iter.next();
            if input.is_none() || label.is_none() {
                break;
            }
            let input = input.unwrap();
            let label = label.unwrap();
            // now get input and label as slices
            let input = input.as_slice();
            let label = label.as_slice();
            // now assign the input and label to the tensor
            for i in 0..input.len() {
                input_tensor[i] = input[i].clone();
            }
            for i in 0..label.len() {
                label_tensor[i] = label[i].clone();
            }

            let mut run_args = SessionRunArgs::new();
            run_args.add_target(&self.minimize);

            let error_squared_fetch = run_args.request_fetch(&self.Error, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;
            // do the above prints in one line
            println!(
                "training on {}\n input: {:?} label: {:?} error: {}",
                i, input, label, res
            );

            result.push(res);
        }

        //TODO: save the model
        Ok(result)
    }
}

//TODO: scrape this for loading a graph
// fn eval<P: AsRef<Path>>(save_dir: P) -> Result<(), Box<dyn Error>> {
//     //load the graph
//     let mut graph = Graph::new();
//     let bundle = SavedModelBundle::load(
//         &SessionOptions::new(),
//         &["serve", "train"],
//         &mut graph,
//         save_dir,
//     )?;
//     let session = &bundle.session;
//     let signature = bundle.meta_graph_def().get_signature(REGRESS_METHOD_NAME)?;
//     let input_info = signature.get_input(REGRESS_INPUTS)?;
//     let output_info = signature.get_output(REGRESS_OUTPUTS)?;
//     let input_op = graph.operation_by_name_required(&input_info.name().name)?;
//     let output_op = graph.operation_by_name_required(&output_info.name().name)?;

//     let mut input_tensor = Tensor::<f32>::new(&[1, 2]);
//     for i in 0..4 {
//         input_tensor[0] = (i & 1) as f32;
//         input_tensor[1] = ((i >> 1) & 1) as f32;
//         let expected = ((i & 1) ^ ((i >> 1) & 1)) as f32;
//         let mut run_args = SessionRunArgs::new();
//         run_args.add_feed(&input_op, input_info.name().index, &input_tensor);
//         let output_fetch = run_args.request_fetch(&output_op, output_info.name().index);
//         session.run(&mut run_args)?;
//         let output = run_args.fetch::<f32>(output_fetch)?[0];
//         let error = (output - expected) * (output - expected);
//         println!("Error: {}", error);
//         if error > 0.1 {
//             return Err(Box::new(Status::new_set(
//                 Code::Internal,
//                 &format!("Error too high: {}", error),
//             )?));
//         }
//     }

//     Ok(())
// }

#[cfg(test)]
mod tests {
    #[test]
    fn test_all() {
        println!("test_all");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut scope = Scope::new_root_scope();
        let norm_net = NormNet::new(&mut scope, 2, 1, 10, 10, 10, 0.001, 5 as f32).unwrap();

        //FITNESS FUNCTION//
        //TODO: pass in dyn fitness function instead of hardcoded in class?

        //TRAIN//
        let mut rrng = rand::thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..100000 {
            // instead of the above, generate either 0 or 1 and cast to f32
            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

            inputs.push(input);
            outputs.push(output);
        }

        assert_eq!(inputs.len(), outputs.len());
        norm_net.train(inputs, outputs).unwrap();
    }
}
