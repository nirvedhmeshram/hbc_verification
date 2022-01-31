# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Callable, Tuple

import click
import torch
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    # pyre-ignore[21]
    from fbgemm_gpu import open_source  # noqa: F401
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


def benchmark_hbc_function(
    func: Callable[[Tensor], Tuple[Tensor, Tensor]],
    input: Tensor,
) -> Tuple[float, Tensor]:
    if input.is_cuda:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        # Benchmark code
        output1, output2 = func(input)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output1, output2 = func(input)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output1, output2


@click.command()
@click.option("--iters", default=1)
@click.option("--warmup-runs", default=0)
def main(
    iters: int,
    warmup_runs: int,
) -> None:

    data_types = [torch.float]

    total_time = {
        "hbc_by_feature": {
            "cpu": {
                torch.float: 0.0,
            },
        },
    }

    num_bins: int = 5000
    num_segments: int = 42

    num_logits = 5000
    input_data_cpu = torch.rand(num_logits, dtype=torch.float)

    segment_lengths: Tensor = torch.randint(0, 2, (num_logits,))
    #segment_lengths: Tensor = torch.ones(num_logits).long()
    #segment_offsets: Tensor = torch.randint(0, 2, (num_logits,))
    segment_offsets: Tensor = torch.cumsum(segment_lengths,0)
    segment_offsets: Tensor = torch.cat((torch.tensor([0]),segment_offsets),0)
    num_values: int = int(torch.sum(segment_lengths).item())
    segment_values: Tensor = torch.randint(
        0,
        num_segments,
        (num_values,),
    )

    lower_bound: float = 0.0
    upper_bound: float = 1.0
    w: float = (upper_bound - lower_bound) / num_bins

    bin_num_examples: Tensor = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
    bin_num_positives: Tensor = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
    bin_boundaries: Tensor = torch.arange(
        lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
    )

    by_feature_bin_num_examples: Tensor = torch.empty(
        [num_bins * (num_segments + 1)], dtype=torch.float64
    ).fill_(0.0)
    by_feature_bin_num_positives: Tensor = torch.empty(
        [num_bins * (num_segments + 1)], dtype=torch.float64
    ).fill_(0.0)
    
    d = {'segment_lengths': segment_lengths, 'segment_offsets': segment_offsets, 'logits':input_data_cpu,'segment_values':segment_values}
    torch.save(d, 'hbc_inputs.pt')

    def fbgemm_hbc_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration(
            input,
            bin_num_examples,
            bin_num_positives,
            0.4,
            lower_bound,
            upper_bound,
            0,
            0.9995,
        )

    def fbgemm_hbc_by_feature_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            input,
            segment_values,
            segment_lengths,
            num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            num_bins,
            0.4,
            lower_bound,
            upper_bound,
            0,
            0.9995,
        )
        
    def fbgemm_generic_hbc_by_feature_cpu(input: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            input,
            segment_values,
            segment_lengths,
            num_segments,
            by_feature_bin_num_examples,
            by_feature_bin_num_positives,
            bin_boundaries,
            0.4,
            0,
            0.9995,
        )

    for step in range(iters + warmup_runs):
        for data_type in data_types:
            curr_input = input_data_cpu.to(data_type)
            hbc_by_feature_time, feature_output1, feature_output2 = benchmark_hbc_function(
                fbgemm_hbc_by_feature_cpu, curr_input
            )

            d = {'feature_output1': feature_output1, 'feature_output2': feature_output2}
            torch.save(d, 'hbc_outputs.pt')
            if step >= warmup_runs:
                total_time["hbc_by_feature"]["cpu"][data_type] += hbc_by_feature_time

    for op, curr_items in total_time.items():
        for platform, data_items in curr_items.items():
            for dtype, t_time in data_items.items():
                logging.info(
                    f"{op}_{platform}_{dtype} time per iter: {t_time / iters * 1.0e6:.0f}us"
                )


if __name__ == "__main__":
    main()
