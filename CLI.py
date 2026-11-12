import stats_in_python as stats_m
import os
import sys

def main():
    print("Load file"+"-"*20)
    print("1. Data_file")
    print("2. Data_file2")
    print("3. Data_file3")
    selection = input("Which file will be loaded? ")
    files_dir = "./files"
    if selection == "1":
        filepath = os.path.join(files_dir, "data_file.txt")
    elif selection == "2":
        filepath = os.path.join(files_dir, "data_file2.txt")
    elif selection == "3":
        filepath = os.path.join(files_dir, "data_file3.txt")
    else:
        print("Non-valid selection")
        sys.exit()
    
    column = int(input("Select the column: "))
    
    numbers, total_values = stats_m.load_data(filepath, column)
    print("Count:", total_values)
    print("ValidNum:", len(numbers))
    print("Average", stats_m.calculate_average(numbers))
    max_val, min_val = stats_m.get_max_and_min(numbers)
    print("Maximum:", max_val)
    print("Minimum:", min_val)
    print("Variance:", stats_m.calculate_variance(numbers))
    print("Std Dev:", stats_m.calculate_std_dev(numbers))
    print("Median:", stats_m.calculate_median(numbers))    


if __name__ == "__main__":
    main()
