
# inicialização
library(caret)
library(keras)
library(dplyr)




























# carrega base de dados
fullDatabase <- read.csv2(file = "exemplo.csv")

View(head(fullDatabase))

fullDatabase$ANO <- as.character(fullDatabase$ANO)
fullDatabase$MES <- as.character(fullDatabase$MES)
fullDatabase$DIA <- as.character(fullDatabase$DIA)
fullDatabase$HORA <- as.character(fullDatabase$HORA)




















# realiza o feature hashing
featureHashing <- dummyVars(~ ANO + MES + DIA + HORA + factor(DIA_DA_SEMANA), data = fullDatabase, fullRank = T)
featureHashingDF <- data.frame(predict(featureHashing, newdata = fullDatabase))

View(head(featureHashingDF))

# renomeia as colunas da base alterada
colNames <- colnames(featureHashingDF)
colNames[67:72] <- c('DIA_DA_SEMANA_2', 'DIA_DA_SEMANA_3', 'DIA_DA_SEMANA_4', 'DIA_DA_SEMANA_5', 'DIA_DA_SEMANA_6', 'DIA_DA_SEMANA_7')
colnames(featureHashingDF) <- colNames

# agrega as informações na base
fullDatabaseHashed <- cbind(fullDatabase, featureHashingDF)

View(head(fullDatabaseHashed))



















# função para normalização dos dados
normalized <- function(y) {
  x <- y[!is.na(y)]
  x <- (x - min(x)) / (max(x) - min(x))
  y[!is.na(y)] <- x
  return(y)
}

normalizedCol <- apply(fullDatabaseHashed[, c("QTD", "QTD_NEGADA", "QTD_NEG_OP", "QTD_APROVADA", "QTD_CREDITOR_DISTINCT")], 2, normalized)
colnames(normalizedCol) <- c('QTD_NORM', 'QTD_NEGADA_NORM', 'QTD_NEG_OP_NORM', 'QTD_APROVADA_NORM', 'QTD_CREDITOR_DISTINCT_NORM')

View(head(normalizedCol))

# agrega as informações na base
fullDatabaseHashedNormalized <- data.frame(cbind(fullDatabaseHashed, normalizedCol))

View(head(fullDatabaseHashedNormalized))

















# cria as bases de treino, validação e teste
training <- (fullDatabaseHashedNormalized %>% filter(paste0('20', ANO, '-', MES, '-', DIA) <= '2017-06-30'))[, 11:87]
validation <- (fullDatabaseHashedNormalized %>% filter(paste0('20', ANO, '-', MES, '-', DIA) >= '2017-07-01' & 
                                                         paste0('20', ANO, '-', MES, '-', DIA) <= '2018-01-31'))[, 11:87]
test <- (fullDatabaseHashedNormalized %>% filter(paste0('20', ANO, '-', MES, '-', DIA) >= '2018-02-01' & paste0('20', ANO, '-', MES, '-', DIA) <= '2018-04-30'))[, 11:87]






























# define uma seed
set.seed(7)
use_session_with_seed(7)

# cria a instância do modelo
model <- keras_model_sequential()

# cria as três camadas do modelo
model %>% 
  layer_dense(units = 64, activation = 'tanh', input_shape = 73) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'relu')

# define a função de custo com método de otimização Adam
model %>% 
  compile(loss = "mean_squared_error", 
          optimizer = "adam")

# salva o modelo
checkpoint <- callback_model_checkpoint(
  filepath = "model_64x64x1_tanh_relu_100_32.hdf5",
  save_best_only = TRUE,
  period = 1,
  verbose = 1
)

# define critério de parada
early_stopping <- callback_early_stopping(patience = 10)

# salva o log do TensorBoard (visualização de gráficos dinâmicos)
tf_board <- callback_tensorboard("log_model_64x64x1_tanh_relu_100_32")

# roda o modelo efetivamente
history <- model %>%
  fit(x = as.matrix(training[, c(1:72, 77)]),
      y = as.matrix(training[, 73]),
      epochs = 100,
      batch_size = 32,
      validation_data = list(as.matrix(validation[, c(1:72, 77)]), as.matrix(validation[, 73])),
      callbacks = list(checkpoint, early_stopping, tf_board))

View(head(training[, c(1:72, 77)]))
View(head(training[, 73]))

# salva o histórico das iterações
saveRDS(history, "history_model_64x64x1_tanh_relu_100_32.hdf5.RDS")

# exibe o resultado das iterações
plot(history)








