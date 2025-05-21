# S2SBench 
[📄 View Paper](https://arxiv.org/abs/2505.14438)  [📥 Download Dataset](https://huggingface.co/datasets/undobug/S2SBench)

S2SBench is a benchmark designed to evaluate the intelligence degradation of speech-to-speech large language models.



## The Dataset

S2SBench includes three evaluation sets:

- **sStoryCloze**: English speech-based story cloze task.
- **zh-sStoryCloze**: Chinese speech-based story cloze task.
- **sCMMLU**: Speech-based version of CMMLU, covering multiple-choice questions across various disciplines.

### Dataset Statistics

| Dataset         | Sample Pairs | Positive per Pair | Negative per Pair |
|-----------------|--------------|-------------------|-------------------|
| sStoryCloze     | 3742         | 1                 | 1                 |
| zh-sStoryCloze  | 3742         | 1                 | 1                 |
| sCMMLU          | 4743         | 1                 | 3                 |

---

## Evaluating a Customized Model

This section explains how to evaluate your own model on S2SBench. Two evaluation modes are provided: speech-to-text and text-to-text.

### Speech-to-Text Evaluation

To evaluate your model in the **speech-to-text** setting:

```sh
cd s2t
bash bash.sh
```

### Text-to-Text Evaluation
```sh
cd t2t
bash bash.sh
```

## About `bash.sh` Script

Here is an example of the `s2t/bash.sh` script:

```bash
# Run the inference script with dataset list and plotting enabled
python s2t_infer_ppl.py --dataset_list sStory_s2t zh_story cmmlu_write_4 --plot
```
