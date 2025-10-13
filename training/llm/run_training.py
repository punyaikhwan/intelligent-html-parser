from flan_t5_training import HTMLParsingTrainer

def main():
    trainer = HTMLParsingTrainer()
    model_trainer = trainer.train(
        train_file="training_data.json",
        output_dir="./flan-t5-html-parser-2",
        num_epochs=5,
        batch_size=4,
        learning_rate=3e-4,
        save_steps=100,
        eval_steps=100
    )
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
