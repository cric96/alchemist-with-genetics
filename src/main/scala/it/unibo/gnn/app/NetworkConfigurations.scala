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
  val stateSize = 3
  val featureSize = 1
  val edgeSize = 1
  val outputSize = 1
  // Linear networks shapes
  val biasConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize).nOut(3).build(),
      new DenseLayer.Builder().nIn(3).nOut(stateSize).build()
    )
    .build()

  val neighbourStateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(featureSize + edgeSize + featureSize).nOut(5).build(),
      new DenseLayer.Builder().nIn(5).nOut(stateSize * stateSize).build()
    )
    .build()
  // Non linear network shapes
  val stateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize + edgeSize + featureSize).nOut(40).build(),
      new DenseLayer.Builder().nIn(40).nOut(20).build(),
      new DenseLayer.Builder().nIn(20).nOut(stateSize).build()
    ).build()

  val aggregationConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(stateSize * 2).nOut(50).build(),
      new DenseLayer.Builder().nIn(50).nOut(25).build(),
      new DenseLayer.Builder().nIn(25).nOut(stateSize).build()
    ).build()

  // Common output shape
  val outputConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize).nOut(20).name("layer--0").build(),
      new DenseLayer.Builder().nIn(20).nOut(10).name("layer--1").build(),
      new DenseLayer.Builder().nIn(10).nOut(outputSize).name("layer--2").build()
    ).build()

  val linearConfig : NetworkConfiguration = NetworkConfiguration(
    stateSize,
    featureSize,
    edgeSize,
    outputSize,
    LinearGNNCodec(biasConfiguration, neighbourStateConfiguration, outputConfiguration, maxWeight = 2),
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
