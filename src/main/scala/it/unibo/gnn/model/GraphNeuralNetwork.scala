package it.unibo.gnn.model

import it.unibo.gnn.model.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.jdk.CollectionConverters.ListHasAsScala

trait GraphNeuralNetwork {
  def eval(feature : INDArray, neighborhood: List[NeighborhoodData]) : NodeState
}

object GraphNeuralNetwork {
  case class NonLinearGraphNeuralNetwork(stateEvolution : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork) extends GraphNeuralNetwork {
    private val layers = stateEvolution.getLayerWiseConfigurations
      .getConfs
      .asScala
      .map(_.getLayer)
      .collect { case dense : DenseLayer => dense}
    private val stateSize = layers.reverse.head.getNOut
    def eval(feature : INDArray, neighborhood: List[NeighborhoodData]) : NodeState = {
      val state = neighborhood.map(neighbor => stateEvolution.output(neighbor.concatWithNodeFeature(feature), false))
        .foldRight[INDArray](new NDArray(1, stateSize))((acc, data) => acc.addi(data))
      val outputNetworkInput : INDArray = new NDArray(state :: feature :: Nil map { _.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
      val result = outputEvaluation.output(outputNetworkInput, false)
      NodeState(state, result)
    }
  }
  case class LinearGraphNeuralNetwork(biasNetwork : MultiLayerNetwork, neighbourNetwork : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork, mu : Double) extends GraphNeuralNetwork {
    def eval(feature : INDArray, neighborhood: List[NeighborhoodData]) : NodeState = {
      val local = biasNetwork.output(feature)
      val stateDimension = local.length()
      val factor = mu / (stateDimension) * neighborhood.size
      val neighbourEvaluation = neighborhood.map(data => (data.state, data.concatWithNodeFeature(feature)))
        .map { case (neighbourState, data) => (neighbourState, neighbourNetwork.output(data))}
        .map { case (neighbourState, matrix) => (neighbourState, matrix.reshape(stateDimension, stateDimension)) }
        .map { case (neighbourState, matrix) => (neighbourState, matrix) }
        .map { case (xn, w) => w.mulRowVector(xn) }
        .map { array => array.mul(factor) }
        .foldRight[INDArray](new NDArray(1, stateDimension))((acc, data) => acc.addi(data))
      val aggregation = outputEvaluation.output(neighbourEvaluation, false)
      val state = aggregation.sum(local)
      val output = outputEvaluation.output(state, false)
      NodeState(state, output)
    }
  }
  //Data model
  case class NeighborhoodData(feature : INDArray, state : INDArray, edgeFeature : INDArray) {
    def concatWithNodeFeature(nodeFeature : INDArray) : INDArray = {
      new NDArray(nodeFeature :: feature :: state :: edgeFeature :: Nil map { node => node.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
    }
    def concatFeatureAndEdge(nodeFeature : INDArray) : INDArray = {
      new NDArray(nodeFeature :: feature :: edgeFeature :: Nil map { node => node.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
    }
  }
  case class NodeState(state : INDArray, output : INDArray)
}
