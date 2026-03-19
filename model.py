import pickle

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

    def summary(self):
        total_params = 0
        total_memory = 0
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            param_count = 0
            mem = 0
            if hasattr(layer, "W"):
                param_count += layer.W.size + layer.b.size
                mem += layer.W.nbytes + layer.b.nbytes
            total_params += param_count
            total_memory += mem
            print(f" Layer {i}: {layer_name} — Params: {param_count}, Memory: {mem / 1024:.2f} KB")
        print(f"Total Parameters: {total_params}")
        print(f"Total Estimated Memory: {total_memory / 1024:.2f} KB")

    def save(self, filename):
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                all_params.append(layer.get_params())
            else:
                all_params.append(None)
        with open(filename, 'wb') as f:
            pickle.dump(all_params, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            all_params = pickle.load(f)
        for layer, params in zip(self.layers, all_params):
            if hasattr(layer, 'set_params') and params is not None:
                layer.set_params(params)