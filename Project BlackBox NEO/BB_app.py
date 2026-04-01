import sys
import BB_engine

def print_header():
    print("=" * 55)
    print("      NEO Black Box Baseline")
    print("=" * 55)

def main_menu():
    print("\nLoading dataset and preparing environment")
    data = BB_engine.prepare_data()
    if data is None:
        return
    print("Data loaded successfully\n")
    
    while True:
        print("\n" + "-"*30)
        print("          MAIN MENU")
        print("-" * 30)
        print("1. Train New Black Box Model")
        print("2. Evaluate Saved Model")
        print("3. Predict MOID for Specific NEO")
        print("4. Visualize Top Threat Orbits")
        print("0. Exit")
        print("-" * 30)
        
        choice = input("Select an option (0-4): ").strip()
        
        if choice == '1':
            epochs_input = input("Enter number of epochs (default 5000): ").strip()
            epochs = int(epochs_input) if epochs_input.isdigit() else 5000
            BB_engine.train_model(data, epochs=epochs)
            
        elif choice == '2':
            BB_engine.evaluate_model(data)
            
        elif choice == '3':
            name_query = input("Enter NEO name (e.g., 'Asteroid-12'): ").strip()
            if name_query:
                BB_engine.predict_single(data, name_query)
                
        elif choice == '4':
            top_n_input = input("How many NEOs to visualize? (default 10): ").strip()
            top_n = int(top_n_input) if top_n_input.isdigit() else 10
            BB_engine.generate_visualization(data, top_n=top_n)
            
        elif choice == '0':
            print("Exiting program.")
            sys.exit(0)
            
        else:
            print("Invalid choice. Please select a number from 0 to 4.")

if __name__ == "__main__":
    print_header()
    main_menu()
