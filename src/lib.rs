//! Deep reinforcment learning algorithm for Rattus to predict the location of the next click and with what probability.
//! uses (i32,i32,f32) where the first i32 tuple is the location in x and y coordinates and f32 is the confidence probability of the click from 0-1.

//TODO: allow this to fallback to TF-CPU for machines without GPU and possibly just do inference with a build script.
//TODO: floats stink norm net is an attempt to realizing rot net and uint8 operations with bounded functions while retaining all the flaws in DNNs..

//allow unstable features
// #![feature(int_log)]
use half::bf16;
use half::f16;
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

///construct a graph scope for building, trainning and evaluating an ANN.
fn test_train<P: AsRef<Path>>(save_dir: P) -> Result<(), Box<dyn Error>> {
    // ================
    // Build the model.
    // ================
    let mut scope = Scope::new_root_scope();
    let scope = &mut scope;

    //construct a norm_net with scope
    let (layers, variables, input, label, scope) = norm_net(scope, 2, 1, 10, 5, 100)?;
    let output = layers.last().unwrap().to_owned();

    //TODO: REFACTOR into method (begin builder)
    //pythagorean distance for error because outputs can be negative
    let error = ops::sqrt(
        ops::pow(
            ops::sub(output.clone(), label.clone(), scope)?,
            ops::constant(2.0 as f32, scope)?,
            scope,
        )?,
        scope,
    )?;
    // let error = ops::sqrt(error, scope)?;
    let error = ops::pow(error.clone(), ops::constant(2.0 as f32, scope)?, scope)?;

    //TODO: NEED MOMENTUM
    // let optimizer =
    // GradientDescentOptimizer::new(ops::constant(0.01f32, scope).unwrap());
    //TODO: there should be a way to treat oscillations (thrashing between set of local minima)
    //      and local minima with momentum now that bias is gone
    let mut optimizer = AdadeltaOptimizer::new();
    optimizer.set_epsilon(ops::constant(1e-5 as f32, scope)?);
    optimizer.set_rho(ops::constant(0.95 as f32, scope)?);
    optimizer.set_learning_rate(ops::constant(0.001 as f32, scope)?);

    let (minimizer_vars, minimize) = optimizer
        .minimize(
            scope,
            error.clone().into(),
            MinimizeOptions::default().with_variables(&variables),
        )?
        .into();

    //TODO: extract into a method
    // ===================
    // Saved Model Builder
    // ===================
    let mut all_vars = variables.clone();
    all_vars.extend_from_slice(&minimizer_vars);
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
                        name: input.name()?,
                        //TODO: what is this index in terms of the output tensor?
                        index: 0,
                    },
                ),
            );
            def.add_output_info(
                REGRESS_OUTPUTS.to_string(),
                TensorInfo::new(DataType::Float, Shape::from(None), output.name()?),
            );
            def
        });
    let saved_model_saver = builder.inject(scope)?;

    //TODO: seperate this from construction in case of towers etc.
    //      pass in a scope and return a scope.

    // =========================
    // Initialize the variables.
    // =========================
    let options = SessionOptions::new();
    let g = scope.graph_mut();
    let session = Session::new(&options, &g)?;
    let mut run_args = SessionRunArgs::new();
    // Initialize variables we defined.
    for var in &variables {
        run_args.add_target(&var.initializer());
    }
    // Initialize variables the optimizer defined.
    for var in &minimizer_vars {
        run_args.add_target(&var.initializer());
    }
    session.run(&mut run_args)?;
    //TODO: END REFACTOR
    // pass in train functionally

    // ================
    // Train the model.
    // ================
    let mut input_tensor = Tensor::new(&[1, 2]);
    let mut label_tensor = Tensor::new(&[1, 1]);
    // the trainning routine
    // TODO: pass this in as a lambda for virtual interface
    let mut train = |i| -> Result<Tensor<f32>, Box<dyn Error>> {
        //generate a random number between -1 and 1 for input tensor
        //regress on multiplication surface for -1 > x > 1
        let mut rrng = rand::thread_rng();

        // //NOTE: I ran out of trivial test cases to run so I did all of them
        //MOD
        // let c = rrng.gen_range(-1.0..1.0);
        // let d = rrng.gen_range(-1.0..1.0);
        // input_tensor[2] = f16::from_f32(c);
        // input_tensor[3] = f16::from_f32(d);
        // label_tensor[1] = f16::from_f32(c % d);

        //XOR
        input_tensor[0] = ((i & 1) as f32);
        input_tensor[1] = (((i >> 1) & 1) as f32);
        label_tensor[0] = (((i & 1) ^ ((i >> 1) & 1)) as f32);

        //DIV
        // let a = rrng.gen_range(-1.0..1.0);
        // let b = rrng.gen_range(-1.0..1.0);
        // input_tensor[0] = a as f32;
        // input_tensor[1] = b as f32;

        // label_tensor[0] = (a / b) as f32;

        let mut run_args = SessionRunArgs::new();
        //print the output of layer5
        // run_args.add_target(&layer1_output);
        run_args.add_target(&minimize);
        let error_squared_fetch = run_args.request_fetch(&error, 0);
        // let layer5_output_fetch = run_args.request_fetch(&layer1_output, 0);
        run_args.add_feed(&input, 0, &input_tensor);
        run_args.add_feed(&label, 0, &label_tensor);
        session.run(&mut run_args)?;
        //print layer5 output
        if i % 1000 == 0 {
            println!("\n ----SIGNALS----\n");
            // println!("{}", run_args.fetch::<f16>(layer5_output_fetch)?);
            println!("\n\n");
        }
        Ok(run_args.fetch(error_squared_fetch)?)
    };

    let mut j: usize = 0;
    let mut i = 0;
    loop {
        if j == 1000000 {
            // if j == 10000{
            break;
        }
        j = j + 1;
        if i == i32::max_value() {
            i = 0;
        } else {
            i = i + 1;
        }
        // for i in 0..10000 {
        // if j % 10 == 0 {
        //print 4 iterations
        println!("{} with {}", j, train(i)?);
        println!("{} with {}", j, train(i + 1)?);
        println!("{} with {}", j, train(i + 2)?);
        println!("{} with {}", j, train(i + 3)?);
        println!("");
        // } else {
        //     train(i)?;
        // }
    }

    // ================
    // Save the model.
    // ================
    // saved_model_saver.save(&session, &g, &save_dir)?;

    // ===================
    // Evaluate the model.
    // ===================
    // for i in 0..4 {
    //     let error = train(i)?;
    //     println!("Error: {}", error);
    //     if error > 0.1 {
    //         return Err(Box::new(Status::new_set(
    //             Code::Internal,
    //             &format!("Error too high: {}", error),
    //         )?));
    //     }
    // }
    //TODO: print the connection weights for each layer
    // ================
    // Print the connection weights.
    // ================
    // println!("{}", run_args.fetch::<f32>(value)?[0]);

    Ok(())
}

// BEGIN CLASS REWORK
//TODO: deprecate the above functional code once the stateful code is feature complete and tested
//TODO: save and load from disk
/// a Normalizing Network currently being researched
/// NOTE: currently inputs and outputs must be flattened if representing >1 dim data
struct NormNet<'a> {
    scope: &'a mut Scope,
    session_options: SessionOptions,
    net_layers: Vec<Output>,
    net_vars: Vec<Variable>,
    Input: Operation,
    Label: Operation,
    Output: Output,
}
impl<'a> NormNet<'a> {
    fn new(
        scope: &'a mut Scope,
        input_size: u64,
        output_size: u64,
        layer_width: u64,
        layer_height: u64,
        max_integer: u32,
    ) -> Result<NormNet, Status> {
        // create a norm net with norm_net function above
        // create a scope
        //TODO: can this operate with other TF scopes external to NormNet?
        //      does this overallocate and prevent xla optimizations?
        // let mut scope = Scope::new_root_scope();
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
        //END OF CONSTRUCTING NETWORK

        let Output = net_layers.last().unwrap().clone();

        let options = SessionOptions::new();

        //TODO: sort these
        Ok(NormNet {
            net_layers,
            net_vars,
            Input,
            Label,
            Output,
            scope,
            session_options: options,
        })
    }
    ///train the network with the given inputs and labels (must be synchronized in index order)
    ///PARAMETERS:
    /// inputs: the inputs to the network as a flattened 1D vector
    /// labels: the labels for the inputs as a flattened 1D vector, must align index wise with inputs. (e.g. inputs[0] must be labeled as labels[0])
    /// error: the error operation to minimize with (e.g. pythagorean error sqrt(x^2 + y^2))
    /// learning_rate: the learning rate for the network (e.g. 0.001)
    fn train<T: tensorflow::TensorType>(
        // &mut self,
        self,
        inputs: Vec<&[T]>,
        labels: Vec<&[T]>,
        error: Operation,
        learning_rate: f32,
    ) -> Result<Vec<Tensor<f32>>, Status> {
        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(ops::constant(learning_rate, self.scope)?);

        let (minimize_vars, minimize) = optimizer
            .minimize(
                self.scope,
                error.clone().into(),
                // MinimizeOptions::default().with_variables(&net.1),
                MinimizeOptions::default().with_variables(&self.net_vars),
            )?
            .into();

        let g = self.scope.graph_mut();
        let session = Session::new(&self.session_options, &g)?;

        //TODO: shouldnt need to initialize so much ITL
        //TODO: randomize input and labels and k-fold
        //TODO: create dataset structure
        let mut result = vec![];

        let mut input_tensor: Tensor<T> = Tensor::new(&[1u64, inputs[0].len() as u64]);
        // print inputs[0].len();
        println!("{}", inputs[0].len());
        let mut label_tensor: Tensor<T> = Tensor::new(&[1u64, labels[0].len() as u64]);
        println!("{}", labels[0].len());
        // let layer5_output_fetch = run_args.request_fetch(&layer1_output, 0);
        for (input, label) in inputs.clone().into_iter().zip(labels.clone().into_iter()) {
            println!("{:?}", input);
            println!("{:?}", label);
            let mut run_args = SessionRunArgs::new();
            for var in &self.net_vars {
                run_args.add_target(&var.initializer());
            }
            for var in &minimize_vars {
                run_args.add_target(&var.initializer());
            }
            run_args.add_target(&minimize);
            // session.run(&mut run_args)?;

            let error_squared_fetch = run_args.request_fetch(&error, 0);
            // load input into input tensor
            //TODO: pop off each input and label
            input
                .iter()
                .enumerate()
                .for_each(|x| input_tensor[x.0] = x.1.clone());
            label
                .iter()
                .enumerate()
                .for_each(|x| label_tensor[x.0] = x.1.clone());
            println!("input shape: {:?}", input_tensor.shape());
            println!("label shape: {:?}", label_tensor.shape());

            // let res = run_args.fetch(error_squared_fetch)?;
            let error_squared_fetch = run_args.request_fetch(&error, 0);
            run_args.add_feed(&self.Input, 0, &input_tensor);
            run_args.add_feed(&self.Label, 0, &label_tensor);
            session.run(&mut run_args)?;

            // result.push(run_args.fetch(error_squared_fetch)?);
            result.push(run_args.fetch(error_squared_fetch)?);
        }
        Ok(result)
        //print layer5 output
        // if i % 1000 == 0 {
        //     println!("\n ----SIGNALS----\n");
        //     // println!("{}", run_args.fetch::<f16>(layer5_output_fetch)?);
        //     println!("\n\n");
        // }
        // let error = run_args.fetch(error_squared_fetch)?;
    }
}
// END OF CLASS REWORK

fn eval<P: AsRef<Path>>(save_dir: P) -> Result<(), Box<dyn Error>> {
    //load the graph
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(
        &SessionOptions::new(),
        &["serve", "train"],
        &mut graph,
        save_dir,
    )?;
    let session = &bundle.session;
    let signature = bundle.meta_graph_def().get_signature(REGRESS_METHOD_NAME)?;
    let input_info = signature.get_input(REGRESS_INPUTS)?;
    let output_info = signature.get_output(REGRESS_OUTPUTS)?;
    let input_op = graph.operation_by_name_required(&input_info.name().name)?;
    let output_op = graph.operation_by_name_required(&output_info.name().name)?;

    let mut input_tensor = Tensor::<f32>::new(&[1, 2]);
    for i in 0..4 {
        input_tensor[0] = (i & 1) as f32;
        input_tensor[1] = ((i >> 1) & 1) as f32;
        let expected = ((i & 1) ^ ((i >> 1) & 1)) as f32;
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&input_op, input_info.name().index, &input_tensor);
        let output_fetch = run_args.request_fetch(&output_op, output_info.name().index);
        session.run(&mut run_args)?;
        let output = run_args.fetch::<f32>(output_fetch)?[0];
        let error = (output - expected) * (output - expected);
        println!("Error: {}", error);
        if error > 0.1 {
            return Err(Box::new(Status::new_set(
                Code::Internal,
                &format!("Error too high: {}", error),
            )?));
        }
    }

    Ok(())
}

pub fn test_main() -> Result<(), Box<dyn Error>> {
    let mut dir = env::temp_dir();
    dir.push("tf-rust-example-xor-saved-model");
    let mut dir2 = env::temp_dir();
    dir2.push("tf-rust-example-xor-saved-model2");
    match fs::remove_dir_all(&dir) {
        Err(e) => {
            if e.kind() != ErrorKind::NotFound {
                return Err(Box::new(e));
            }
        }
        Ok(_) => (),
    }
    match fs::remove_dir_all(&dir2) {
        Err(e) => {
            if e.kind() != ErrorKind::NotFound {
                return Err(Box::new(e));
            }
        }
        Ok(_) => (),
    }
    test_train(&dir)?;
    // Ensure that the saved model works even when moved.
    // Users do not need to do this; this is purely for testing purposes.
    fs::rename(&dir, &dir2)?;
    eval(&dir2)?;
    Ok(())
}
//TODO: macro for building norm_net: pass in
//      **args that impl Sized for input
//      rectangular dimensions of the network
//      **args that are type that impl Sized for output
//      fitness function as dyn Fn() -> f32
//      activation function for hidden layers
//      (input and output are tanh and this should default to sigmoid, probably shouldnt change must be symmetric)

//TODO: doc tests are preferable to anything else except integration tests as working examples.
//      unittests must be feature justified and sufficiently complex.

#[cfg(test)]
mod tests {
    #[test]
    fn test_all() {
        println!("test_all");
        //call the main function
        use crate::*;

        //CONSTRUCTION//
        let mut scope = Scope::new_root_scope();
        let norm_net = NormNet::new(&mut scope, 2, 1, 2, 2, 10).unwrap();
        //pythagorean distance for error because outputs can be negative
        // TODO: just one scope
        let error = ops::sqrt(
            ops::pow(
                ops::sub(
                    norm_net.Output.clone(),
                    norm_net.Label.clone(),
                    norm_net.scope,
                )
                .unwrap(),
                ops::constant(2.0 as f32, norm_net.scope).unwrap(),
                norm_net.scope,
            )
            .unwrap(),
            norm_net.scope,
        )
        .unwrap();
        // let error = ops::sqrt(error, scope)?;
        let error = ops::pow(
            error.clone(),
            ops::constant(2.0 as f32, norm_net.scope).unwrap(),
            norm_net.scope,
        )
        .unwrap();

        //TRAIN//
        let mut rrng = rand::thread_rng();
        // create an input and output for NormNet using the above XOR generator
        // let inputs: Vec<&[_]> = vec![&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]];
        // same as above but with f32 instead of double
        let inputs: Vec<&[_]> = vec![
            &[0.0 as f32, 0.0 as f32],
            &[0.0 as f32, 1.0 as f32],
            &[1.0 as f32, 0.0 as f32],
            &[1.0 as f32, 1.0 as f32],
        ];
        // let outputs: Vec<&[_]> = vec![&[0.0], &[1.0], &[1.0], &[0.0]];
        let outputs: Vec<&[_]> = vec![&[0.0 as f32], &[1.0 as f32], &[1.0 as f32], &[0.0 as f32]];
        assert_eq!(inputs.len(), outputs.len());
        norm_net.train(inputs, outputs, error, 0.001).unwrap();
    }
}

// TODO: make this a lib
// create a main function
// pub fn main() {
//     //call the main function
//     use crate::*;

//     let res = test_main();
//     println!("message {:?}", res);
//     //assert res is Ok:
// }
