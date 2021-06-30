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
  val stateSize : 2 = 2
  val featureSize : 1 = 1
  val edgeSize : 1 = 1
  val outputSize : 1 = 1

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
  val stateConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.TANH)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize + edgeSize + featureSize).nOut(7).build(),
      new DenseLayer.Builder().nIn(7).nOut(3).build(),
      new DenseLayer.Builder().nIn(3).nOut(stateSize).build()
    ).build()

  val outputConfiguration: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
    .activation(Activation.RELU)
    .list(
      new DenseLayer.Builder().nIn(stateSize + featureSize).nOut(3).name("layer--0").build(),
      new DenseLayer.Builder().nIn(3).nOut(2).name("layer--1").build(),
      new DenseLayer.Builder().nIn(2).nOut(outputSize).name("layer--2").build()
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
    NonLinearGNNCodec(stateConfiguration, outputConfiguration, maxWeight = 2),
    "result-non-linear.xml"
  )
}
