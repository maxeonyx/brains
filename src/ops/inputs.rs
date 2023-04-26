


pub struct Input {
    name: String,
    sig: (DType, Dims),
}

pub struct Inputs {
    default: Option<Input>,
    rest: HashSet<Input>,
}
