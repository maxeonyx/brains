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

///organization struct to handle the paradigm for serialized sessions and graphs (feature disparity in Rust Tensorflow).
struct Serialized {
    //TODO: non-clonable shared reference movement
    // SavedModel: SavedModelBundle,
    session: Session,
    graph: Graph,
    // signature: &'a SignatureDef,
    input_info: TensorInfo,
    input: Operation,
    output_info: TensorInfo,
    output: Operation,
    label_info: TensorInfo,
    label: Operation,
    error_info: TensorInfo,
    error: Operation,
    minimize_info: TensorInfo,
    minimize: Operation,
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
pub struct NormNet<'a> {
    ///the scope for tensorflow to prevent having multiple scopes active
    scope: &'a mut Scope,
    ///Session options currently being used
    session: Session,
    session_options: SessionOptions,
    // graph: Option<Graph>,
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
    Serialized: Option<Serialized>,
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
        assert!(max_integer % 10 == 0, "max_integer must be a multiple of 10 since it represents order of magnitude of the integer range");

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
            &mut scope.with_op_name("error"),
        )
        .unwrap();

        let (minimize_vars, minimize) = optimizer
            .minimize(
                scope,
                Error.clone().into(),
                MinimizeOptions::default().with_variables(&net_vars),
            )?
            .into();
        let session = Session::new(&options, &mut scope.graph())?;

        // TODO: initialize session and graph here?

        Ok(NormNet {
            scope,
            session: session,
            session_options: options,
            //TODO: does this need to be declared or can we always self.scope.graph()? may conflict if user is using scope for other things
            // graph: None,
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
            Serialized: None,
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
            let mut builder = tensorflow::SavedModelBuilder::new();
            builder
                .add_collection("train_vars", &all_vars)
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
                    def.add_output_info(
                        REGRESS_OUTPUTS.to_string(),
                        TensorInfo::new(DataType::Float, Shape::from(None), self.Output.name()?),
                    );

                    def
                });
            let saved_model_saver = builder.inject(self.scope)?;
            self.SavedModelSaver.replace(Some(saved_model_saver));
        }
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
    pub fn load(&'a mut self, dir: String) -> Result<(), Box<dyn Error>> {
        // load the model from disk in the current directory
        println!("loading previously saved model..");
        //TODO: check examples in tensorflow rust for solution:
        // load the model from the saved model saver for this training session
        // let save_dir = std::env::current_dir().unwrap();
        let mut graph = Graph::new();
        //TODO: ensure we can access variables from graph or otherwise
        let bundle = SavedModelBundle::load(
            //TODO: test here that this overwrites correctly
            // &SessionOptions::new(),
            &self.session_options,
            &["serve", "train"],
            &mut graph,
            dir,
        )?;
        //TODO: associate the operations with the class state
        // let session = &bundle.session;
        let signature = bundle
            .meta_graph_def()
            .get_signature(REGRESS_METHOD_NAME)?
            .clone();
        //TODO: ensure cloning the bits in the construction below retains the proper graph pointers
        // let input_info = signature.get_input(REGRESS_INPUTS)?;
        // let label_info = signature.get_input("label")?;
        // let error_info = signature.get_input("error")?;
        // let minimize_info = signature.get_input("minimize")?;
        // let output_info = signature.get_output(REGRESS_OUTPUTS)?;
        // let input = graph.operation_by_name_required(&input_info.name().name)?;
        // let label = graph.operation_by_name_required(&label_info.name().name)?;
        // let error = graph.operation_by_name_required(&error_info.name().name)?;
        // let minimize = graph.operation_by_name_required(&minimize_info.name().name)?;
        // let output = graph.operation_by_name_required(&output_info.name().name)?;
        // now associate the operations with the class state by creating Some(Serialized)
        self.Serialized = Some(Serialized {
            // SavedModel: bundle,
            session: bundle.session,
            // signature: &signature,
            input_info: signature.get_input(REGRESS_INPUTS)?.clone(),
            input: graph
                .operation_by_name_required(&signature.get_input(REGRESS_INPUTS)?.name().name)?,
            output_info: signature.get_output(REGRESS_OUTPUTS)?.clone(),
            output: graph
                .operation_by_name_required(&signature.get_output(REGRESS_OUTPUTS)?.name().name)?,
            label_info: signature.get_input("label")?.clone(),
            label: graph.operation_by_name_required(&signature.get_input("label")?.name().name)?,
            error_info: signature.get_input("error")?.clone(),
            error: graph.operation_by_name_required(&signature.get_input("error")?.name().name)?,
            minimize_info: signature.get_input("minimize")?.clone(),
            minimize: graph
                .operation_by_name_required(&signature.get_input("minimize")?.name().name)?,
            //NOTE: shouldnt need to interact with the graph after this, ablate this due to abstraction once tested
            graph: graph,
        });
        // set the variables to be the loaded variables
        Ok(())
    }
    ///TODO: pub fn serialize(self, uuid) // serialize NormNet including the session-graph to disk

    //TODO:
    ///train the network with theGiven inputs and labelswhich must be
    /// synchronized in index order initializes a saved model saver if not already initialized and saves after training.
    // pub fn train_checkpoint<T: tensorflow::TensorType>(
    //     self,
    //     inputs: Vec<Vec<T>>,
    //     labels: Vec<Vec<T>>,
    //     // TODO: pass these is instead of constructing
    //     // error: Operation,
    //     // learning_rate: f32,
    // ) -> Result<Vec<Tensor<f32>>, Status> {
    // }

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
        // TODO: pass these is instead of constructing
        // error: Operation,
        // learning_rate: f32,
    ) -> Result<Vec<Tensor<f32>>, Status> {
        //TODO: allow to change learning rate and error here
        // load the model if it has been saved otherwise initialize a saved model saver
        //TODO: extract this so we can call it in other training methods as a "refresh" serialization synchronization
        // TODO: also a sub method for save() and load()

        let g = self.scope.graph();
        let mut run_args = SessionRunArgs::new();

        // set parameters to be optimization targets
        for var in &self.net_vars {
            run_args.add_target(&var.initializer());
        }
        for var in &self.minimize_vars {
            run_args.add_target(&var.initializer());
        }
        self.session.run(&mut run_args)?;

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
            self.session.run(&mut run_args)?;

            let res: Tensor<f32> = run_args.fetch(error_squared_fetch)?;

            // do the above prints in one line
            println!(
                "training on {}\n input: {:?} label: {:?} error: {}",
                i, input, label, res
            );

            result.push(res);
        }

        //TODO: save the model
        //if self.session is none save it
        // if self.session.is_none() {
        // self.session = Some(session);
        // }
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
        //TODO: once session is in constructor remove the optionals and this
        let mut norm_net =
            NormNet::new(&mut scope, 2, 1, 10, 10, 10, 0.0000000000001, 5 as f32).unwrap();

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

    //TODO: save/load unittest
    #[test]
    fn test_serialization() {
        println!("test_serialization");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut scope = Scope::new_root_scope();
        let mut norm_net =
            NormNet::new(&mut scope, 2, 1, 10, 10, 10, 0.0000000000001, 5 as f32).unwrap();
        //TRAIN//
        let mut rrng = rand::thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        // create 100 entries for inputs and outputs of xor
        for _ in 0..10 {
            // instead of the above, generate either 0 or 1 and cast to f32
            let input = vec![(rrng.gen::<u8>() & 1) as f32, (rrng.gen::<u8>() & 1) as f32];
            let output = vec![(input[0] as u8 ^ input[1] as u8) as f32];

            inputs.push(input);
            outputs.push(output);
        }

        assert_eq!(inputs.len(), outputs.len());
        norm_net.train(inputs, outputs).unwrap();

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
    }
}
