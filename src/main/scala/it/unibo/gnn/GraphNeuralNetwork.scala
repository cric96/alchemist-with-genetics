package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.jdk.CollectionConverters.ListHasAsScala
case class GraphNeuralNetwork(stateEvolution : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork) {
  private val layers = stateEvolution.getLayerWiseConfigurations.getConfs.asScala.map(_.getLayer)
  private val stateSize = layers.collect { case layer : DenseLayer => layer }.head.getNOut
  def eval(feature : INDArray, neighborhood: List[NeighborhoodData]) : NodeState = {
    val state = neighborhood.map(neighbor => stateEvolution.output(neighbor.concatWithNodeFeature(feature), false))
      .foldRight[INDArray](new NDArray(1, stateSize))((acc, data) => acc.addi(data))
    val outputNetworkInput : INDArray = new NDArray(state :: feature :: Nil map { _.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
    val result = outputEvaluation.output(outputNetworkInput, false)
    NodeState(state, result)
  }
}
object GraphNeuralNetwork {
  case class NeighborhoodData(feature : INDArray, state : INDArray, edgeFeature : INDArray) {
    def concatWithNodeFeature(nodeFeature : INDArray) : INDArray = {
      new NDArray(nodeFeature :: feature :: state :: edgeFeature :: Nil map { node => node.toDoubleVector.map(_.toFloat) } reduce { _ ++ _ })
    }
  }
  case class NodeState(state : INDArray, output : INDArray)
}
