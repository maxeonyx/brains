//TODO: layers that can be passed in to Network in a separate crate to allow architecture engineering
use half::f16;
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

//TODO: need a partial for generic type so these can be passed functionally:

/// A standard fully connected layer with bias term
///
/// `activation` is a function which takes a tensor and applies an activation
/// function such as sigmoid.
///
/// Returns variables created and the layer output.
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
    //TODO: trait macro for extra args or a builder with trait class?
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
