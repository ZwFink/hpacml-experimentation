from dataclasses import dataclass


@dataclass
class BOParameterWrapper:
    parameter_space: list
    parameter_constraints: list
    objectives: dict
    tracking_metric_names: list

    def get_parameter_names(self):
        return [p['name'] for p in self.parameter_space]


@dataclass
class EvalArgs:
    benchmark_name: str
    config_global: dict
    config_filename: str
    arch_parameters: dict
    hyper_parameters: dict
