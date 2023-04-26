use tensorflow::Output;
use std::collections::HashMap;
use maplit::hashmap;

pub struct Outputs {
    default: Option<Output>,
    rest: HashMap<&'static str, Output>
}

#[macro_export]
macro_rules! out {
    ($e:expr) => {
        use brains::ops::Outputs;
        use tensorflow::Output;
        use std::collections::HashMap;
        use maplit::hashmap;
        Outputs {
            default: Some($e),
            rest: HashMap::new(),
        }
    };
    { default => $e:expr, $($x:tt)*} => {
        use brains::ops::Outputs;
        use tensorflow::Output;
        use std::collections::HashMap;
        use maplit::hashmap;
        Outputs {
            default: Some($e),
            rest: hashmap!{$($x)*}
        }
    };
    { default $name:literal => $e:expr, $($x:tt)*} => {
        use brains::ops::Outputs;
        use tensorflow::Output;
        use std::collections::HashMap;
        use maplit::hashmap;
        let default = $e;
        Outputs {
            default: Some(default),
            rest: hashmap!{
                $name => default,
                $($x)*
            },
        }
    };
    ($($x:tt)*) => {
        use brains::ops::Outputs;
        use tensorflow::Output;
        use std::collections::HashMap;
        use maplit::hashmap;
        Outputs {
            default: None,
            rest: hashmap!{$($x)*}
        }
    };
}
pub use out;
