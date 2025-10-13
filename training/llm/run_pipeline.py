from flan_t5_training import HTMLParsingTrainer

def main():
    trainer = HTMLParsingTrainer()
    model_trainer = trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
