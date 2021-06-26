package it.unibo.gnn

import it.unibo.gnn.evolutionary.GNNCodec
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation

object NetworkConfiguration {
  val stateConfiguration : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(7).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(4).build()
    ).build()

  val outputConfiguration : MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(5).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(4).build(),
      new DenseLayer.Builder().nIn(4).nOut(1).build()
    ).build()
  val codec : GNNCodec = GNNCodec(stateConfiguration, outputConfiguration, maxWeight = 2)
}
