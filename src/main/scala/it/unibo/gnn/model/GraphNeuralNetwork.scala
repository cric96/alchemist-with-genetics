package it.unibo.gnn.model

import it.unibo.gnn.model.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import it.unibo.gnn.model.NDArrayUtils._
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.jdk.CollectionConverters.ListHasAsScala

trait GraphNeuralNetwork {
  def eval(feature : INDArray, neighborhood: Seq[NeighborhoodData]) : NodeState
}

object GraphNeuralNetwork {
  case class NonLinearGraphNeuralNetwork(stateEvolution : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork) extends GraphNeuralNetwork {
    private val layers = stateEvolution.getLayerWiseConfigurations
      .getConfs
      .asScala
      .map(_.getLayer)
      .collect { case dense : DenseLayer => dense}
    private val stateSize = layers.reverse.head.getNOut
    private val zero = ndarray((0 until stateSize.toInt).map(_ => 0f))
    def eval(feature : INDArray, neighborhood: Seq[NeighborhoodData]) : NodeState = {
      val state = neighborhood.map(neighbor => stateEvolution.output(neighbor.concatWithNodeFeature(feature), false))
        .reduceOption((acc, data) => acc.add(data))
        .map(_.div(neighborhood.size))
        .getOrElse(zero)

      val outputNetworkInput : INDArray = ndarray(state :: feature :: Nil map { _.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
      val result = outputEvaluation.output(outputNetworkInput, false)
      NodeState(state, result)
    }
  }
  case class LinearGraphNeuralNetwork(biasNetwork : MultiLayerNetwork, neighbourNetwork : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork, mu : Double) extends GraphNeuralNetwork {
    def eval(feature : INDArray, neighborhood: Seq[NeighborhoodData]) : NodeState = {
      val local = biasNetwork.output(feature)
      val stateDimension = local.length()
      val factor = mu / stateDimension * neighborhood.size
      val neighbourEvaluation = neighborhood.map(data => (data.state, data.concatFeatureAndEdge(feature)))
        .map { case (neighbourState, data) => (neighbourState, neighbourNetwork.output(data, false))}
        .map { case (neighbourState, matrix) => (neighbourState, matrix.reshape(stateDimension, stateDimension)) }
        .map { case (neighbourState, matrix) => (neighbourState, matrix) }
        .map { case (xn, w) => w.mmul(xn.transpose()).transpose() }
        .map { array => array.mul(factor) }
        .reduceOption((acc, data) => acc.add(data))
        .getOrElse(Nd4j.zeros(1, stateDimension))
      val state = neighbourEvaluation.add(local)
      val size = concat(feature, state)
      val output = outputEvaluation.output(size, false)
      NodeState(state, output)
    }
  }
  //Data model
  case class NeighborhoodData(feature : INDArray, state : INDArray, edgeFeature : INDArray) {
    def concatWithNodeFeature(nodeFeature : INDArray) : INDArray = concat(nodeFeature, feature, state, edgeFeature)
    def concatFeatureAndEdge(nodeFeature : INDArray) : INDArray = concat(nodeFeature, feature, edgeFeature)
  }
  case class NodeState(state : INDArray, output : INDArray)

}
