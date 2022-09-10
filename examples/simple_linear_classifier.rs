use std::collections::HashMap;

use maplit::hashmap;

struct Outputs {
    default: Option<Tensor>,
    rest: HashMap<&'static str, Tensor>
}

macro_rules! out {
    ($e:expr) => {
        Outputs {
            default: $e,
            rest: HashMap::new(),
        }
    };
    { default => $e:expr, $($x:tt)*} => {
        Outputs {
            default: Output,
            rest: hashmap!{$($x)*}
        }
    };
    { default $name:literal => $e:expr, $($x:tt)*} => {
        Outputs {
            default: Output,
            rest: hashmap!{
                $name => $e,
                $($x)*
            },
        }
    };
    ($($x:tt)*) => {
        Outputs {
            default: None,
            rest: hashmap!{$($x)*}
        }
    };
}

fn main() {

    let B = 32;
    let F = 784;
    let C = 10;

    let model = brains::model("linear_classifier", |tf, input| {

        let w = tf.var::<f32>([F, C]);
        let b = tf.var::<f32>([F]);

        let x = input.required("x", [F]);

        let logits = x.matmul(w);
        let probabilities = logits.softmax();
        let y = probabilities.argmax();

        out! {
            default "y" => y,
            "logits" => logits,
            "probs" => probabilities,
        }
    });

    let loss = brains::function("categorical_nll_loss", |tf, input| {

        let logits = input.required("logits", [C]);
        let targ = input.required("targ", [C]);
        
        let mult_logits = targ * logits + (1 - targ) * (1 - logits);
        let loss = - mult_logits.sum();
        
        out! { loss }
    });

    let trainer = brains::TrainingLoop::new(model, loss)
        .with_optimizer(brains::SGD::default())


}
