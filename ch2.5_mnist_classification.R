#renv::install("keras")
library(tensorflow)
library(keras)
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_images <- train_images / 255
test_images <- mnist$test$x
test_images <- test_images / 255
train_labels <- mnist$train$y
test_labels <- mnist$test$y

model <- keras_model_sequential(input_shape = c(28, 28)) %>%
    layer_flatten() %>%
    layer_dense(128, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(10)

loss_fn <- loss_sparse_categorical_crossentropy(from_logits = TRUE)
model |>
    compile(
        optimizer = "rmsprop",
        loss = loss_fn,
        metrics = "accuracy"
    )

fit(model,train_images, train_labels, epochs = 5, batch_size = 128)
model |>  evaluate(test_images, test_labels, verbose = 2)
predictions <- predict(model, test_images)

#reticulate::virtualenv_install(env="r-tensorflow",packages = "pandas")
pd <- reticulate::import("pandas")
df <- pd$DataFrame(tensorflow::tf$nn$softmax(predictions))

library(tidyverse)
pred_df <- df |> as_tibble()
colnames(pred_df) <- paste0("Digit",sort(unique(test_labels)))
pred_df$ID <- 1:nrow(pred_df)
pred_df |>
    pivot_longer(
        cols = starts_with("Digit"),
        names_to = "Label",
        values_to = ".pred"
    ) |>
    mutate(Label = factor(Label,levels=paste0("Digit",sort(unique(test_labels))))) |>
    filter(ID==1) |>
    ggplot(aes(Label,.pred,group=ID)) +
    geom_line() +
    scale_y_continuous(labels = scales::percent) +
    labs(x=NULL,y="Probability",title="First Testing Image Prediction") +
    theme_bw(base_size = 16)
