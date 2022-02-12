//! NormNet: a Deep Artificial Neural Network using normalizing neurons to investigate alternatives to bias
//! based expressivity. Gradient based connection-wise dropout is also included due to the numeric properties
//! inherent to normalization (thanks to a little trig and calculus). This allows NormNet to also be a sparse-searching
//! "top down" architecture search where the parameters are recoverable thanks again to the gradient.
//! It is hypothesized and investigated here that the aforementioned features will lead to much more robust transfer
//! learning and therein generalization. The only significant downsides to this is a division by constant operation on each node
//! (seemingly/empirically not as bad as it sounds).

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
use rand::Rng;
use std::os;
use std::rc::Rc;
use std::time::{Duration, Instant};
use tensorflow::ops;
use tensorflow::train::AdadeltaOptimizer;
use tensorflow::train::GradientDescentOptimizer;
use tensorflow::train::MinimizeOptions;
use tensorflow::train::Optimizer;
use tensorflow::BFloat16;
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
use uuid::Uuid;

//TODO: this may be able to be abstracted into a high level NN crate that allows architecture development by
//      passing layer in functionally then using the high level abstractions of NormNet (rename to ANN or something)
//      build a seperate library of network architectures like tf.NN

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

pub fn norm_layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Operation, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output, Operation), Status> {
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
    let output_op = ops::div(
        ops::mat_mul(
            input,
            //this sets the gradients to dropout weights
            ops::tan(w.output().clone(), scope)?,
            scope,
        )?,
        n,
        scope,
    )?;

    let act = activation(
        //this normalizes to speed up trainning and sample efficiency
        output_op.into(),
        scope,
    )?;
    let output_op = act.clone();
    // .into(); //,
    Ok((vec![w.clone()], act.into(), output_op))
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

//TODO: rename this now that methods are introduced and abstractions have occured but keep for low level accessibility
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
    let (vars, layer, _) = norm_layer(
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
        let (vars, layer, _) = norm_layer(
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
    let (vars, output, _) = norm_layer(
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
//================
//----NORM_NET----
//================
/// A standard fully connected layer without bias trainnable parameters
/// instead normalizing and dropping out connections at each node.
///
/// NOTE: currently inputs and outputs must be flattened if representing >1 dim data
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
/// * ~shouldn't~ have exploding gradient although vanishing gradient may be possible due to normalizing division.
///      
/// * technically connection wise dropout is divisive (top down) architecture search (e.g.: the opposite of NEAT (bottom up) which is agglomerative)
///
/// CONS:
/// * may be slower due to more operations (division)
///       
/// * input and output must/should be tailored for normalized input/output (standard data science practices)
///       
/// * needs large type precision for stability, but the stability can be tuned (as apposed to bias which needs architectural considerations)
///
/// #
/// NOTE: parameters goes to zero whereas biases find some 1-dimensional partition from -inf to inf. This helps
/// build subgraph search modules (subtrees essentially). That can quickly optimize for distinct domains and labels via dropout.
///
/// NOTE:
/// Tanh should the first and last layers activation function to map inputs and outputs to negative values.
/// multiplying the output by some multiple of 10 allows the otherwise normalized network to take in and output whole integers.
/// -x > y > x | x > 1
///
/// NOTE:
/// We dont use BFloat since the integer range is only used as a buffer for addition overflow in matmul.
/// In all other operations we are strictly bounded -1 > x > 1. As long as layer_width is not
/// greater than Float range we are fine in the worst case (summing all 1's).
/// Otherwise decimal precision of float type is our parameter type precision.
pub struct NormNet {
    /// Tensorflow objects for user abstraction from Tensorflow
    scope: Scope,
    session: Session,
    session_options: SessionOptions,
    ///each layers output for the network
    net_layers: Vec<Output>,
    ///all the trainable parameters of the network
    net_vars: Vec<Variable>,
    ///Operations to interact with the graph (kept here for serialization)
    Input: Operation,
    Label: Operation,
    Output_op: Operation,
    Error: Operation,
    ///variables to be minimized
    minimize_vars: Vec<Variable>,
    ///regression operation
    minimize: Operation,
    ///class for serializing, saving and loading the model
    SavedModelSaver: RefCell<Option<SavedModelSaver>>,
    is_loaded: bool,
}
impl NormNet {
    pub fn new(
        input_size: u64,
        output_size: u64,
        layer_width: u64,
        layer_height: u64,
        max_integer: u32,
        learning_rate: f32,
        error_power: f32,
    ) -> Result<NormNet, Status> {
        assert!(max_integer % 10 == 0 || max_integer == 1, "max_integer must be a multiple of 10 or 1 since it represents order of magnitude of the integer range");
        let mut scope = Scope::new_root_scope();

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
        let (vars, layer, _) = norm_layer(
            Input.clone(),
            input_size,
            layer_width,
            &|x, scope| Ok(ops::tanh(x, scope)?.into()),
            &mut scope,
        )?;
        net_vars.extend(vars);
        net_layers.push(layer.clone());

        let mut prev_layer = layer;
        //hidden layers
        for i in 0..layer_height - 2 {
            let (vars, layer, _) = norm_layer(
                prev_layer.clone(),
                layer_width,
                layer_width,
                //NOTE: originally designed with tan but vanishing gradient can occur
                &|x, scope| Ok(ops::tanh(x, scope)?.into()),
                &mut scope,
            )?;
            prev_layer = layer.clone();

            net_vars.extend(vars);
            net_layers.push(layer.clone());
        }

        //the final output layer is tanh to express negative values and multiplied to stabilize the
        //half precision gradient as well as express whole integers outside of -1 and 1.
        let (vars, output, Output_op) = norm_layer(
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
            &mut scope,
        )?;
        net_vars.extend(vars);
        net_layers.push(output.clone());
        //END OF CONSTRUCTING NETWORK

        // let Output = net_layers.last().unwrap().to_owned();
        let Output = output;

        let options = SessionOptions::new();
        let init_saved_model_saver = RefCell::new(None);

        //TODO: this needs to be in state so that it can be saved for transfer learning
        //  let mut optimizer = AdadeltaOptimizer::new();
        //  optimizer.set_learning_rate(ops::constant(learning_rate, &mut scope)?);
        let mut optimizer =
            GradientDescentOptimizer::new(ops::constant(learning_rate, &mut scope)?);

        // DEFINE ERROR FUNCTION //
        //TODO: pass this in conditionally, give user output and label with
        //      a partial constructor then they supply error and construction is complete
        // two structs? whats standard functionally for partial construction?
        //default error is pythagorean distance
        let Error = ops::sqrt(
            ops::pow(
                ops::sub(Output.clone(), Label.clone(), &mut scope)?,
                ops::constant(2.0 as f32, &mut scope)?,
                &mut scope,
            )?,
            &mut scope,
        )?;
        //TODO: at least pass in the constant here as a gradient weighting scalar coefficient
        //TODO: scalar coeffecient instead to prevent curving the gradient
        let Error = ops::pow(
            Error.clone(),
            ops::constant(error_power, &mut scope).unwrap(),
            &mut scope.with_op_name("error"),
        )
        .unwrap();

        let (minimize_vars, minimize) = optimizer
            .minimize(
                &mut scope,
                Error.clone().into(),
                MinimizeOptions::default().with_variables(&net_vars),
            )?
            .into();
        let session = Session::new(&options, &mut scope.graph())?;

        // set parameters to be optimization targets if they havent been set already
        let mut run_args = SessionRunArgs::new();
        //TODO: This is a strange way to initialize variables
        for var in &net_vars {
            run_args.add_target(&var.initializer());
        }
        for var in &minimize_vars {
            run_args.add_target(&var.initializer());
        }
        session.run(&mut run_args)?;

        Ok(NormNet {
            scope,
            session: session,
            session_options: options,
            net_layers,
            net_vars,
            Input,
            Label,
            Output_op,
            Error,
            minimize_vars,
            minimize,
            SavedModelSaver: init_saved_model_saver,
            is_loaded: false,
        })
    }

    // forward pass and return result
    // fn infer
    //one time traning that can take output of infer and a label
    // fn backprop

    //save the model out to disk in this directory as default
    pub fn save(&mut self, dir: String) -> Result<(), Box<dyn Error>> {
        // save the model to disk in the current directory
        if self.SavedModelSaver.borrow().is_none() {
            //TODO: this should be in constructor but is here currently because borrow checker problems
            // if self.session.is_none() {
            // let mut session_options = &self.session_options;
            // let session = Session::new(&session_options, &self.scope.graph())?;
            // let session = self.session;
            // self.session = Some(session);
            // }
            println!("initializing saved model saver..");
            let mut all_vars = self.net_vars.clone();
            all_vars.extend_from_slice(&self.minimize_vars);
            let mut last_layer = self.net_layers.last().unwrap().clone();
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
                    def.add_input_info(
                        "label".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Label.name()?,
                                index: 0,
                            },
                        ),
                    );
                    //NOTE: we only need this for reporting learning position for learning rate
                    def.add_input_info(
                        "error".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.Error.name()?,
                                index: 0,
                            },
                        ),
                    );
                    def.add_input_info(
                        "minimize".to_string(),
                        TensorInfo::new(
                            DataType::Float,
                            Shape::from(None),
                            OutputName {
                                name: self.minimize.name()?,
                                index: 0,
                            },
                        ),
                    );
                    // def.add_output_info(
                    //     REGRESS_OUTPUTS.to_string(),
                    //     TensorInfo::new(DataType::Float, Shape::from(None), self.Output.name()?),
                    // );

                    def
                });
            let saved_model_saver = builder.inject(&mut self.scope)?;
            self.SavedModelSaver.replace(Some(saved_model_saver));
        }
        //TODO: annotate this with user input, fitness and checkpoint number
        // generate a uuid
        let uuid = Uuid::new_v4();
        self.SavedModelSaver.borrow_mut().as_mut().unwrap().save(
            &self.session,
            // &self.graph.unwrap(),
            &self.scope.graph(),
            //TODO: ensure this is cur_directory and pass in user specification with this as default
            format!("{}/{}", dir, uuid),
        )?;
        Ok(())
    }

    ///load the model from disk and store it in self.Serialized
    pub fn load(&mut self, dir: String) -> Result<(), Box<dyn Error>> {
        // load the model from disk in the given directory
        println!("loading previously saved model..");
        let mut graph = Graph::new();
        //TODO: ensure we can access variables from graph or otherwise
        let bundle =
            SavedModelBundle::load(&self.session_options, &["serve", "train"], &mut graph, dir)?;
        let signature = bundle
            .meta_graph_def()
            .get_signature(REGRESS_METHOD_NAME)?
            .clone();

        self.session = bundle.session;
        self.Input =
            graph.operation_by_name_required(&signature.get_input("inputs")?.name().name)?;
        self.Label =
            graph.operation_by_name_required(&signature.get_input("label")?.name().name)?;
        self.Error =
            graph.operation_by_name_required(&signature.get_input("error")?.name().name)?;
        self.minimize =
            graph.operation_by_name_required(&signature.get_input("minimize")?.name().name)?;

        // TODO: @DEPRECATED with construction var initialization
        self.is_loaded = true;
        Ok(())
    }

    /// train the network with the given inputs and labels (must be synchronized in index order)
    ///PARAMETERS:
    /// inputs: the inputs to the network as a flattened 1D vector
    /// labels: the labels for the inputs as a flattened 1D vector, must align index wise with inputs. (e.g. inputs[0] must be labeled as labels[0])
    /// error: the error operation to minimize with (e.g. pythagorean error sqrt(x^2 + y^2))
    /// learning_rate: the learning rate for the network (e.g. 0.001)
    pub fn train<T: tensorflow::TensorType>(
        // &mut self,
        &mut self,
        inputs: Vec<Vec<T>>,
        labels: Vec<Vec<T>>,
        // TODO: extract from class: pass these is instead of constructing
        // error: Operation,
        // learning_rate: f32,
    ) -> Result<Vec<Tensor<f32>>, Status> {
        let mut run_args = SessionRunArgs::new();


        //TODO: randomize input and labels while k-folding
        println!("trainning..");
        let mut result = vec![];

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);

        println!("inputs.len(): {}", inputs.len());
        println!("{}", inputs[0].len());
        println!("{}", labels[0].len());

        let mut input_iter = inputs.into_iter();
        let mut label_iter = labels.into_iter();
        input_iter.next();
        label_iter.next();
        let mut i = 0;
        let mut avg_t = vec![];
        loop {
            // start a timer
            let start = Instant::now();

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
            let output = run_args.request_fetch(&self.Output_op, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;
            let output: Tensor<T> = run_args.fetch(output)?;

            // get how long has passed
            let elapsed = start.elapsed();
            avg_t.push(elapsed.as_secs_f32());

            // update the moving average for time
            let average = avg_t.iter().sum::<f32>() / avg_t.len() as f32;

            println!(
                "training on {}\n input: {:?} label: {:?} error: {} output: {} time/epoch(ms): {:?}",
                i, input, label, res, output,average 
            );

            result.push(res);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_net() {
        println!("test_net");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut scope = Scope::new_root_scope();
        let mut norm_net = NormNet::new(2, 1, 10, 10, 10, 10.0, 5 as f32).unwrap();

        //FITNESS FUNCTION//
        //TODO: auto gen labels from outputs and fitness function.

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

    //TODO: save/load unittest
    #[test]
    fn test_serialization() {
        println!("test_serialization");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut scope = Scope::new_root_scope();
        let mut norm_net = NormNet::new(2, 1, 20, 15, 10, 0.01, 5 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..700 {
            // instead of the above, generate either 0 or 1 and cast to f32
            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

            inputs.push(input);
            outputs.push(output);
        }

        assert_eq!(inputs.len(), outputs.len());
        norm_net.train(inputs.clone(), outputs.clone()).unwrap();

        // save the network
        norm_net
            .save("test_serialized_models/".to_string())
            .unwrap();

        //load the network
        let mut path = "";
        //NOTE: dont ever call a string something else in your crates or someone I know will find you.
        let mut e = std::path::PathBuf::new();
        for entry in fs::read_dir("test_serialized_models/").unwrap() {
            let entry = entry.unwrap();
            let is_dir = entry.path();
            //ensure e is a directory
            if !is_dir.is_dir() {
                continue;
            } else {
                e = is_dir;
            }
            path = e.to_str().unwrap();
            println!("{:?}", path);
        }
        let path = path.to_string();
        println!("{:?}", path);

        norm_net.load(path.to_string()).unwrap();

        //TODO: perform all operations and assert the parameters are the same
        norm_net.train(inputs, outputs).unwrap();
    }
}
