import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
def calculate_accuracy_and_average_loss(jsonl_file):
    data_list = []

    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            uttid = data["uttid"]
            loss = data["loss"][0]  
            if isinstance(uttid, int):
                prefix = uttid
            else:
                prefix = uttid.split('_')[0] 
            loss_value = data['loss'][0] if data['loss'] else None  
            data_list.append((prefix, loss, data))
            

    data_list.sort(key=lambda x: int(x[0]))
    loss_even_all = 0
    loss_odd_all = 0
    total_pairs = 0
    correct_num = 0
    for i in range(0, len(data_list), 2):
        if i + 1 < len(data_list):  
            prefix_even, loss_even, item_even = data_list[i]
            prefix_odd, loss_odd, item_odd = data_list[i + 1]
            loss_even_all += loss_even
            loss_odd_all +=loss_odd
            if int(prefix_even) != int(prefix_odd)-1: 
                import pdb;pdb.set_trace()
        
            
            if loss_even < loss_odd:
                correct_num += 1

            total_pairs += 1

 
    accuracy = correct_num / total_pairs if total_pairs > 0 else 0
    loss_even_all = loss_even_all / total_pairs
    loss_odd_all = loss_odd_all / total_pairs
   
    return accuracy,loss_even_all,loss_odd_all          


def process_directory_story(base_dir, output_file):
    results = []

    for subdir in sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and len(d.split('_')) > 2 and d.split('_')[1].isdigit()],key=lambda x: int(x.split('_')[1])):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            jsonl_file = os.path.join(subdir_path, 'result.jsonl')
            if os.path.exists(jsonl_file):
                accuracy, avg_answer_loss, avg_incorrect_loss = calculate_accuracy_and_average_loss(jsonl_file)
                step = int(subdir.split('_')[2])
                results.append((step, accuracy, avg_answer_loss, avg_incorrect_loss))

    # Save results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for step, acc, avg_ans_loss, avg_inc_loss in sorted(results, key=lambda x: x[0]):
            f.write(f"{step},{acc:.2%},{avg_ans_loss:.4f},{avg_inc_loss:.4f}\n")


def plot_and_save_results_from_file(txt_file, output_dir):
    steps = []
    accuracies = []
    avg_answer_losses = []
    avg_incorrect_losses = []


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

  
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            step, acc_str, avg_ans_loss_str, avg_inc_loss_str = line.strip().split(',')
            step = int(step)
            acc = float(acc_str.rstrip('%'))
            avg_ans_loss = float(avg_ans_loss_str)
            avg_inc_loss = float(avg_inc_loss_str)

            steps.append(step)
            accuracies.append(acc / 100)  
            avg_answer_losses.append(avg_ans_loss)
            avg_incorrect_losses.append(avg_inc_loss)


    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Accuracy Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, avg_answer_losses, marker='x', linestyle='-', color='g', label='Average Correct Loss')
    plt.plot(steps, avg_incorrect_losses, marker='s', linestyle='--', color='r', label='Average Incorrect Loss')
    plt.title('Losses Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

  
    plt.tight_layout()

  
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_over_steps.png')
    losses_plot_path = os.path.join(output_dir, 'losses_over_steps.png')

 
    plt.savefig(accuracy_plot_path, bbox_inches='tight')
    plt.close()
    

    plt.figure(figsize=(6, 6))
    plt.plot(steps, avg_answer_losses, marker='x', linestyle='-', color='g', label='Average Correct Loss')
    plt.plot(steps, avg_incorrect_losses, marker='s', linestyle='--', color='r', label='Average Incorrect Loss')
    plt.title('Losses Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(losses_plot_path, bbox_inches='tight')
    plt.close()

    print(f"Plots have been saved to {output_dir}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
 

    result_dir = args.result_dir
    output_file = os.path.join(result_dir, 'results_summary.txt')

    process_directory_story(result_dir, output_file)

    print(f"Results have been saved to {output_file}")
    txt_file = output_file
    output_directory = result_dir  

    plot_and_save_results_from_file(txt_file, output_directory)