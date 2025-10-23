def preprocess_data(image_data, ground_truth, window_size=5):
    # Normalize the image data
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Add spatial information via padding
    padded_image = np.pad(image_data, ((window_size//2, window_size//2),
                                       (window_size//2, window_size//2),
                                       (0, 0)), mode='reflect')
    spatial_spectral_data = np.zeros((image_data.shape[0], image_data.shape[1],
                                      window_size, window_size, image_data.shape[2]))
    for i in range(image_data.shape[0]):
        for j in range(image_data.shape[1]):
            spatial_spectral_data[i, j] = padded_image[i:i+window_size, j:j+window_size, :]

    spatial_spectral_data = spatial_spectral_data.reshape(-1, window_size, window_size, image_data.shape[2])
    y = ground_truth.flatten()
    mask = y != 0
    spatial_spectral_data = spatial_spectral_data[mask]
    y = y[mask]

    # Encode labels from 0 to num_classes-1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return spatial_spectral_data, y, label_encoder

if __name__ == "__main__":
    # File paths (update as needed)
    image_file = "/content/PaviaU.mat"
    gt_file = "/content/PaviaU_gt.mat"

    # Load and preprocess
    image_data, ground_truth = load_pavia_university(image_file, gt_file)
    spatial_spectral_data, y, label_encoder = preprocess_data(image_data, ground_truth)

    # Split data
    train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    train_dataset = PaviaUniversityDataset(spatial_spectral_data[train_indices], y[train_indices])
    test_dataset = PaviaUniversityDataset(spatial_spectral_data[test_indices], y[test_indices])

    # Initialize model
    model = newFastViT(num_channels=103, num_classes=len(np.unique(y)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training setup
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    data_collator = lambda data: {'x': torch.stack([d['x'] for d in data]), 'labels': torch.stack([d['labels'] for d in data])}
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
        data_collator=data_collator
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    # Predictions and metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(f"OA: {overall_accuracy(y[test_indices], y_pred):.4f}")
    print(f"AA: {average_accuracy(y[test_indices], y_pred):.4f}")
    print(f"Kappa: {kappa_coefficient(y[test_indices], y_pred):.4f}")
    f1, precision, recall = calculate_f1_precision_recall(y[test_indices], y_pred)
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Latency and throughput
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Latency per image: {calculate_latency_per_image(model, test_loader, device):.4f} ms")
    print(f"Throughput: {calculate_throughput(model, test_loader, device):.2f} samples/second")

    # Model stats
    print(f"Parameters: {count_model_parameters(model):.2f} M")
    print(f"GFLOPs: {calculate_gflops(model, train_dataset, device):.2f}")

    # Visualization
    cm = confusion_matrix(y[test_indices], y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()