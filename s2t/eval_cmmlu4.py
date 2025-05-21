import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse


def calculate_accuracy_and_average_loss(question_file, loss_file):
    
    questions = {}
    with open(question_file, 'r', encoding='utf-8') as qf:
        for line in qf:
            data = json.loads(line)
            questions[data["uttid"]] = data["answer"]

  
    id_loss = defaultdict(dict)
    with open(loss_file, 'r', encoding='utf-8') as lf:
        for line in lf:
            data = json.loads(line)
            key = data["key"]
            loss_value = data["loss"][0] if data["loss"] else None
            if loss_value is not None:
                id_loss[data["uttid"]][key] = loss_value

 
    correct_count = 0
    total_count = 0
    avg_answer_losses = []
    avg_other_losses = []

    for uttid, answer in questions.items():
        if uttid in id_loss:
            losses = id_loss[uttid]
            if answer in losses:
                for key in losses:
                    if key != answer:
                       
                        if losses[answer] <= losses[key]:
                            correct_count += 1
                        total_count += 1

                        avg_answer_losses.append(losses[answer])
                        avg_other_losses.append(losses[key])
                      
                    
    accuracy = correct_count / total_count if total_count > 0 else 0


    avg_answer_loss = sum(avg_answer_losses) / len(avg_answer_losses) if avg_answer_losses else 0
    avg_other_loss = sum(avg_other_losses) / len(avg_other_losses) if avg_other_losses else 0

    return accuracy, avg_answer_loss, avg_other_loss


def process_directory(base_dir, output_file, question_file):
    results = []

    for subdir in sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and len(d.split('_')) > 2 and d.split('_')[1].isdigit()],
        key=lambda x: int(x.split('_')[1])
    ):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            loss_file = os.path.join(subdir_path, 'result.jsonl')
            if os.path.exists(loss_file):
                step = int(subdir.split('_')[2])
                accuracy, avg_answer_loss, avg_other_loss = calculate_accuracy_and_average_loss(question_file, loss_file)
                results.append((step, accuracy, avg_answer_loss, avg_other_loss))

   
    with open(output_file, 'w', encoding='utf-8') as f:
        for step, acc, avg_ans_loss, avg_other_loss in sorted(results, key=lambda x: x[0]):
            f.write(f"{step},{acc:.2%},{avg_ans_loss:.4f},{avg_other_loss:.4f}\n")


def plot_and_save_results_from_file(txt_file, output_dir):
    steps = []
    accuracies = []
    avg_answer_losses = []
    avg_other_losses = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

   
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            step, acc_str, avg_ans_loss_str, avg_other_loss_str = line.strip().split(',')
            step = int(step)
            acc = float(acc_str.rstrip('%'))
            avg_ans_loss = float(avg_ans_loss_str)
            avg_other_loss = float(avg_other_loss_str)

            steps.append(step)
            accuracies.append(acc / 100)
            avg_answer_losses.append(avg_ans_loss)
            avg_other_losses.append(avg_other_loss)

  
    plt.figure(figsize=(12, 6))


    plt.subplot(1, 2, 1)
    plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(steps, avg_answer_losses, marker='x', linestyle='-', color='g', label='Average Answer Loss')
    plt.plot(steps, avg_other_losses, marker='s', linestyle='--', color='r', label='Average Other Loss')
    plt.title('Losses Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results_plot.png'))
    plt.close()
    print(f"Plots have been saved to {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir")
    parser.add_argument("--question_file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    result_dir = args.result_dir
    question_file = args.question_file
    output_file = os.path.join(result_dir, 'results_summary.txt')

 
    process_directory(result_dir, output_file, question_file)

    print(f"Results have been saved to {output_file}")


    plot_and_save_results_from_file(output_file, result_dir)
