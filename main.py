from python.analysis import analysis

def main():
    model = analysis.run_runners()
    analysis.analyse_results(model, "./raw_data/microbenchmarks_data.json", "./raw_data/kernel_data.json", "./results/results.json")
    return

if __name__ == "__main__":
    main()
