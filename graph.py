import rkgb.src as rkgb
from rkgb.utils.ast_add_on import ast_to_str

# Purpose: parse the rkgb results into a class 
class Graph:
    def __init__(
        self,
        original_model,
        model_inputs,
        verbose=False,
    ):
        super().__init__()
        self.model = original_model
        self.inputs = model_inputs
        self.rkgb_results = rkgb.make_all_graphs(
            original_model, model_inputs, verbose=verbose, bool_kg=True
        )
        self.graph_list = self.rkgb_results.K_graph_list
        self.dict_constants = self.rkgb_results.K_graph_list[0].dict_constants
        self.eq_classes = self.rkgb_results.equivalent_classes
        self.init_code = ast_to_str(self.graph_list[0].init_code)
        self.output = self.graph_list[-1].output_kdn_data

        

