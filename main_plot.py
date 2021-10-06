from pathlib import Path
from util.plot_utils import plot_logs, plot_precision_recall



def main():
    logs_path = Path('output/21_09_2021_with_corss_en_loss')
    logs_paths_list = [logs_path]
    plot_logs(logs_paths_list)

if __name__ == '__main__':
    main()