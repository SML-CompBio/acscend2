## **Pseudocode**

### **1. Model Definitions**
#### **AttentionBlock**
1. **Inputs**: `input_dim`, `hidden_dim`
2. Define a multihead attention block (`nn.MultiheadAttention`) with:
   - `embed_dim = input_dim`
   - `num_heads = 8`
3. Add a fully connected layer to project attention output to `hidden_dim`.
4. **Forward Pass**:
   - Compute attention output using the input `x`.
   - Pass the attention output through the fully connected layer.

#### **Encoder**
1. **Inputs**: `input_dim`, `hidden_dim`, `attention_dim`
2. Add a linear layer to project `input_dim` to `2048`.
3. Use the `AttentionBlock` to apply attention to the output.
4. Add another fully connected layer to project `hidden_dim` to `512`.
5. Add a dropout layer with probability `0.1`.
6. **Forward Pass**:
   - Pass input through the first linear layer and apply ReLU.
   - Apply attention on the result.
   - Pass through the second linear layer with ReLU.
   - Apply dropout.

#### **Decoder**
1. **Inputs**: `input_dim`, `hidden_dim`, `output_dim`, `signature_matrix`
2. Add a linear layer to project `512` to `1024`.
3. Add another linear layer to project to `output_dim` (cell fractions output).
4. Define `gep_matrix` as a learnable parameter (`nn.Parameter`).
5. Store the `signature_matrix`.
6. **Forward Pass**:
   - Pass input through the linear layer and ReLU.
   - Compute `cell_fractions`:
     - Normalize values between `0-1` using min-max normalization.
     - Normalize rows to sum to 1.
   - Compute `reconstructed_pseudobulk`:
     - Use matrix multiplication between `cell_fractions` and `signature_matrix`.
   - Return `cell_fractions`, `reconstructed_pseudobulk`, and `gep_matrix`.

#### **DeconvolutionModel1**
1. Combine `Encoder` and `Decoder` components.
2. **Forward Pass**:
   - Pass input through the encoder.
   - Pass encoded output through the decoder.

---

### **2. Training Function**
#### **deconvolution_train**
1. **Inputs**: `data`, `sig`, `freq`, `org`, `normalized`
2. Split the data into `train` and `validation` sets using `train_test_split`.
3. Convert inputs into PyTorch tensors (`X_train`, `Y_train`, etc.).
4. Initialize the `DeconvolutionModel1` with required dimensions and `sig_matrix`.
5. Define:
   - Loss function: `MSELoss`
   - Optimizer: `Adam`
6. Set hyperparameters:
   - Learning rate
   - Loss weights (`l1`, `l2`, `l3`, `l4`)
7. **Training Loop**:
   - For each epoch:
     1. Forward pass through the model to get predictions.
     2. Compute losses:
        - Cell fraction loss (`MSELoss`)
        - Pseudobulk reconstruction loss
        - Pseudo-GEP loss
        - GEP-signature loss
        - KL divergence loss
     3. Combine weighted losses into total loss.
     4. Backpropagate and update model parameters.
     5. Evaluate on validation data and calculate:
        - Validation loss
        - CCC and RMSE metrics.
     6. Print progress every 10 epochs.
8. Return the trained model and tensors for further predictions.

---

### **3. Prediction Function**
#### **Deconvoluter**
1. **Inputs**: `data`, `sig`, `freq`, `org`, `normalized`
2. Load the data.
3. Train the model using `deconvolution_train`.
4. Evaluate the model on test data (`X_test_tensor`) to get:
   - `test_cell_fractions`
5. Initialize a second model (`model2`) with adjusted parameters.
6. **Training Loop** (Prediction):
   - Similar to the training loop, with test data.
7. Generate predictions:
   - `gep_predictions1` → Gene expression profile.
   - `test_cell_fractions` → Cell fractions.
8. Return results as Pandas DataFrames.

---

### **4. Predictor Class**
#### **Initialization**
1. Load the pre-trained model (`lr_model.joblib`) and store:
   - Model's feature names
   - Class labels for predictions.

#### **Data Preprocessing**
1. **Input**: Path to CSV data.
2. Load the data using `pandas`.
3. Ensure the columns match the model's features.
4. Perform transformations:
   - Rank data (`rankdata`)
   - Apply `log2` transformation.
   - Standardize (z-score normalization).
5. Return processed data.

#### **Prediction**
1. Predict the class probabilities or labels:
   - If `prob=True`, return probabilities as a DataFrame.
   - If `prob=False`, return predicted class labels.

#### **Call Method**
- Allow calling the class object as a function to run predictions.

---

### **5. Execution Flow**
1. Train the model using the `deconvolution_train` function.
2. Use the `Deconvoluter` function to test and validate predictions.
3. Apply the `Predictor` class for downstream predictions with preprocessed data.
