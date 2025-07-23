# main.py
import training
import testing

def main():
    print("Choose an option:")
    print("1. Train the model")
    print("2. Run real-time detection")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        training.train_model()
    elif choice == '2':
        testing.detect_objects()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
