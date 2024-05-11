#chapter 5 in book and https://tensorflow.rstudio.com/tutorials/keras/overfit_and_underfit
#Question: Does gene expression increase with age?
#renv::install("keras")
library(tensorflow)
library(keras)

kidsides::download_sqlite_db()
conn <- kidsides::connect_sqlite_db()
library(dplyr)
DBI::dbListTables(conn)
drug_by_gene_dataset <-
    tbl(conn,"drug_gene") |>
    collect() |>
    data.table::data.table()
ade_dataset <-
    tbl(conn,'ade_raw') |>
    collect() |>
    data.table::data.table()
drug_dataset <-
    tbl(conn,'drug') |>
    collect() |>
    data.table::data.table()
event_dataset <-
    tbl(conn,'event') |>
    collect() |>
    data.table::data.table()
ade_drug_event_name_dataset <-
    ade_dataset[
        atc_concept_id %in% drug_by_gene_dataset$atc_concept_id
        ][
            drug_dataset,
            on='atc_concept_id'
        ][
            !is.na(safetyreportid)
        ][
            event_dataset,
            on='meddra_concept_id',
            allow.cartesian=TRUE
        ][
            !is.na(safetyreportid) & !is.na(relationship_id_23)
        ]
sex_ade_class_dataset <-
    ade_drug_event_name_dataset |>
    tidyr::pivot_wider(
        id_cols = c(ade,sex),
        names_from = c(meddra_concept_name_4,atc1_concept_name),
        values_from = meddra_concept_code_4,
        values_fn = length,
        values_fill = 0
    )

polypharmacy_ade_class_dataset <-
    ade_drug_event_name_dataset |>
    tidyr::pivot_wider(
        id_cols = c(ade,polypharmacy),
        names_from = c(meddra_concept_name_4,atc1_concept_name),
        values_from = meddra_concept_code_4,
        values_fn = length,
        values_fill = 0
    )

train_data <- polypharmacy_ade_class_dataset |>
    select(-ade,-polypharmacy) |>
    as.matrix()
train_labels <-
    (polypharmacy_ade_class_dataset$polypharmacy>4) |>
    as.integer()

models <- list()
model_histories <- list()
model_metrics_df <- list()
cross_df <- expand.grid(
    model_nds = c(3,16,64),
    dropout_percent = c(0,0.2),
    batch_size= c(512)
)
model_nds <- cross_df$model_nds
dropout_percent <- cross_df$dropout_percent
batch_size <- cross_df$batch_size
n_epochs <- 25
for(i in 1:nrow(cross_df)){
    model_name <- stringr::str_glue(
        '{model_nds[i]}denseunits_',
        '{dropout_percent[i]}dropoutrate_',
        '{batch_size[i]}batch_size'
    ) |> as.character()
    if(dropout_percent[i]==0){
        models[[ model_name ]] <-
            keras_model_sequential() |>
            layer_dense(model_nds[1],activation = 'relu') |>
            layer_dense(model_nds[1],activation = 'relu') |>
            layer_dense(1,activation = 'sigmoid')
    }else{
        models[[ model_name ]] <-
            keras_model_sequential() |>
            layer_dense(model_nds[1],activation = 'relu') |>
            layer_dense(model_nds[1],activation = 'relu') |>
            layer_dropout(dropout_percent[i]) |>
            layer_dense(1,activation = 'sigmoid')

    }
    models[[ model_name ]] |>
        compile(
            optimizer = "rmsprop",
            loss = loss_binary_crossentropy(),
            metrics = "accuracy"
        )
    model_histories[[ model_name ]] <-
        models[[ model_name ]] |>
        fit(train_data, train_labels,
            epochs = n_epochs, batch_size = batch_size[i],
            validation_split = 0.4)

    model_metrics_df[[ model_name ]] <-
        model_histories[[ model_name ]] |>
        purrr::pluck('metrics') |>
        tibble::enframe() |>
        tidyr::unnest(c(value)) |>
        dplyr::mutate(model = model_name,
                      epoch = rep(1:n_epochs,4))
}

library(ggplot2)
model_metrics_df |>
    dplyr::bind_rows() |>
    dplyr::mutate(model = factor(model)) |>
    ggplot(aes(epoch,value)) +
    geom_line(aes(color=name),linewidth=1) +
    geom_point(aes(fill=name),shape=21) +
    scale_color_brewer(palette = 'Set1') +
    scale_fill_brewer(palette = 'Set1') +
    scale_x_continuous(breaks = 1:n_epochs) +
    facet_wrap(~forcats::fct_inorder(model),
               nrow=2)

kidsides::disconnect_sqlite_db(conn)
