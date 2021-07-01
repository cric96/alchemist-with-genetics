package it.unibo.gnn.app

import it.unibo.gnn.evolutionary.GNNCodec
import it.unibo.gnn.evolutionary.GNNCodec._
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.nd4j.linalg.activations.Activation

case class NetworkConfiguration (stateSize : Int,
                                 featureSize : Int,
                                 edgeSize : Int,
                                 outputSize : Int,
                                 codec : GNNCodec,
                                 outputFile : String)
object NetworkConfigurations {
  val stateSize = 5
  val featureSize = 2
  val edgeSize = 1
  val outputSize = 1
  // Linear networks shapes
  val biasConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize).nOut(5).build(),
      new DenseLayer.Builder().nIn(5).nOut(3).build(),
      new DenseLayer.Builder().nIn(3).nOut(stateSize).build()
    )
    .build()

  val neighbourStateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize + edgeSize + featureSize).nOut(7).build(),
      new DenseLayer.Builder().nIn(7).nOut(5).build(),
      new DenseLayer.Builder().nIn(5).nOut(stateSize * stateSize).build()
    )
    .build()
  // Non linear network shapes
  val stateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.SIGMOID)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize + edgeSize + featureSize).nOut(10).build(),
      new DenseLayer.Builder().nIn(10).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(stateSize).build()
    ).build()

  val aggregationConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.SIGMOID)
    .list(
      new DenseLayer.Builder().nIn(stateSize * 2).nOut(16).build(),
      new DenseLayer.Builder().nIn(16).nOut(8).build(),
      new DenseLayer.Builder().nIn(8).nOut(stateSize).build()
    ).build()

  // Common output shape
  val outputConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize).nOut(16).name("layer--0").build(),
      new DenseLayer.Builder().nIn(8).nOut(4).name("layer--1").build(),
      new DenseLayer.Builder().nIn(4).nOut(outputSize).name("layer--2").build()
    ).build()

  val linearConfig : NetworkConfiguration = NetworkConfiguration(
    stateSize,
    featureSize,
    edgeSize,
    outputSize,
    LinearGNNCodec(biasConfiguration, neighbourStateConfiguration, outputConfiguration, maxWeight = 0.9),
    "result-linear.xml"
  )

  val nonLinearConfig : NetworkConfiguration = NetworkConfiguration(
    stateSize,
    featureSize,
    edgeSize,
    outputSize,
    NonLinearGNNCodec(stateConfiguration, aggregationConfiguration, outputConfiguration, maxWeight = 2),
    "result-non-linear.xml"
  )
}
