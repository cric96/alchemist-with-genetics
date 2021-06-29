package it.unibo.gnn.app

import it.unibo.gnn.evolutionary.GNNCodec
import it.unibo.gnn.evolutionary.GNNCodec._
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.nd4j.linalg.activations.Activation

object NetworkConfigurations {
  val stateSize : 4 = 4
  val featureSize : 1 = 1
  val edgeSize : 1 = 1
  val outputSize : 1 = 1

  val biasConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(stateSize).build()
    )
    .build()

  val neighbourStateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize + edgeSize + featureSize).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(stateSize).build()
    )
    .build()
  val stateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize + edgeSize + featureSize).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(6).build(),
      new DenseLayer.Builder().nIn(6).nOut(stateSize).build()
    ).build()

  val outputConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize).nOut(8).name("layer--0").build(),
      new DenseLayer.Builder().nIn(8).nOut(4).name("layer--1").build(),
      new DenseLayer.Builder().nIn(4).nOut(outputSize).name("layer--2").build()
    ).build()

  val nonLinearCodec: GNNCodec = NonLinearGNNCodec(stateConfiguration, outputConfiguration, maxWeight = 2)
  val linearCodec : GNNCodec = LinearGNNCodec.apply(biasConfiguration, neighbourStateConfiguration, outputConfiguration)
  val nonLinearFile = "result-non-linear.xml"
  val linearFile = "result-linear.xml"
}
