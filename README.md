# Adversarial_attacks

# Adversarial Attacks on MNIST with TensorFlow and ART

This repository contains a demonstration of training a simple convolutional neural network on the MNIST dataset and then performing adversarial attacks using the Adversarial Robustness Toolbox (ART).

## Requirements

- Python 3.x
- TensorFlow 2.x
- Adversarial Robustness Toolbox (ART)
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/adversarial-attacks-mnist.git
    cd adversarial-attacks-mnist
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow
    pip install adversarial-robustness-toolbox
    ```

## Running the Code

1. **Import dependencies**:
    ```python
    import tensorflow as tf
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib

    from art.estimators.classification import TensorFlowV2Classifier
    from art.attacks.evasion import FastGradientMethod
    from art.utils import load_dataset
    ```

2. **Increase matplotlib font size**:
    ```python
    matplotlib.rcParams.update({"font.size": 14})
    ```

3. **Load the MNIST dataset**:
    ```python
    (train_images, train_labels), (test_images, test_labels), min, max = load_dataset(name="mnist")
    ```

4. **Create and train a TensorFlow Keras model**:
    ```python
    def create_model():
        model = Sequential([
            Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
            MaxPool2D(pool_size=2),
            Conv2D(filters=64, kernel_size=3, activation="relu"),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(units=128, activation="relu"),
            Dense(units=10, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    model = create_model()
    model.fit(x=train_images, y=train_labels, epochs=10, batch_size=256)
    ```

5. **Define an evasion attack using the Fast Gradient Method**:
    ```python
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    classifier = TensorFlowV2Classifier(clip_values=(0, 1), model=model, nb_classes=10, input_shape=(28, 28, 1), loss_object=loss_object)
    attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.3)
    test_images_adv = attack_fgsm.generate(x=test_images)
    ```

6. **Evaluate the model on clean and adversarial images**:
    ```python
    score_clean = model.evaluate(x=test_images, y=test_labels)
    score_adv = model.evaluate(x=test_images_adv, y=test_labels)
    print(f"Clean test set loss: {score_clean[0]:.2f} vs adversarial set test loss: {score_adv[0]:.2f}")
    print(f"Clean test set accuracy: {score_clean[1]:.2f} vs adversarial set test accuracy: {score_adv[1]:.2f}")
    ```

7. **Visualize the effect of different eps values**:
    ```python
    nrows, ncols = 2, 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))
    eps_to_try = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]
    counter = 0

    for i in range(nrows):
        for j in range(ncols):
            attack_fgsm = FastGradientMethod(estimator=classifier, eps=eps_to_try[counter])
            test_images_adv = attack_fgsm.generate(x=test_images)
            axes[i, j].imshow(test_images_adv[0].squeeze())
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            test_score = classifier._model.evaluate(x=test_images_adv, y=test_labels)[1]
            prediction = np.argmax(model.predict(x=np.expand_dims(test_images_adv[0], axis=0)))
            axes[i, j].set_title(f"Eps value: {eps_to_try[counter]}\nTest accuracy: {test_score * 100:.2f}%\nPrediction: {prediction}")
            counter += 1

    plt.show()
    ```

## Results

The clean test set accuracy is significantly higher than the adversarial test set accuracy, demonstrating the effectiveness of adversarial attacks.
