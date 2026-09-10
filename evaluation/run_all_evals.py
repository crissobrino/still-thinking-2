import main_eval
import advanced_eval
import generate_gold

if __name__ == "__main__":
    #print("gold first")
    #generate_gold.generate_gold()
    print("Starting the full test suite...")
    main_eval.run_main_evaluation()
    print("---")
    advanced_eval.run_advanced_evaluation()
    print("ALL DONE! Check the /evaluation folder")