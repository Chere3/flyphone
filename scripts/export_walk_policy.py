"""Exporta la política de marcha preentrenada de flybody (SavedModel de TF/Sonnet,
figshare 'trained-fly-policies.zip' → policies/walking) a un .npz que
`flyphone/walker.py` ejecuta en numpy, sin TensorFlow ni dm-reverb.

Solo necesita un TensorFlow moderno (hay wheels arm64) para leer el checkpoint:
    uv venv /tmp/tfvenv && uv pip install --python /tmp/tfvenv/bin/python tensorflow
    /tmp/tfvenv/bin/python scripts/export_walk_policy.py <dir_saved_model> flyphone/assets/walk_policy.npz

La red es la DMPO de flybody: batch_concat(obs ordenadas por clave) → Linear(512) →
LayerNorm → tanh → 3×(Linear(512) → ELU) → Linear(59) (media de la gaussiana).
"""
import sys, numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

src, dst = sys.argv[1], sys.argv[2]
sm = saved_model_pb2.SavedModel(); sm.ParseFromString(open(f"{src}/saved_model.pb", "rb").read())
nodes = [n.variable.name for n in sm.meta_graphs[0].object_graph_def.nodes if n.HasField("variable")]
# El checkpoint guarda las variables como _variables/<i> en el mismo orden que el grafo de objetos.
reader = tf.train.load_checkpoint(f"{src}/variables/variables")
vals = [reader.get_tensor(f"_variables/{i}/.ATTRIBUTES/VARIABLE_VALUE") for i in range(len(nodes))]
esperado = ["feedforward_mlp_torso/linear/b", "feedforward_mlp_torso/linear/w",
            "feedforward_mlp_torso/layer_norm/offset", "feedforward_mlp_torso/layer_norm/scale"]
assert nodes[:4] == esperado, nodes
assert nodes[-4:] == ["MultivariateNormalDiagHead/linear/b", "MultivariateNormalDiagHead/linear/w"] * 2, nodes
out = {"in_b": vals[0], "in_w": vals[1], "ln_offset": vals[2], "ln_scale": vals[3],
       "mean_b": vals[-4], "mean_w": vals[-3]}          # la segunda Linear de la cabeza es la escala (softplus); no se usa
hidden = vals[4:-4]; assert len(hidden) % 2 == 0
for i in range(len(hidden) // 2):
    out[f"h{i}_b"], out[f"h{i}_w"] = hidden[2 * i], hidden[2 * i + 1]
out["n_hidden"] = np.array(len(hidden) // 2)
np.savez_compressed(dst, **{k: np.asarray(v, dtype=np.float32) if k != "n_hidden" else v for k, v in out.items()})
print({k: np.shape(v) for k, v in out.items()}, "→", dst)
