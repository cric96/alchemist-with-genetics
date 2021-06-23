package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.{NeighborhoodData, NodeState}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray
case class GraphNeuralNetwork(stateEvolution : MultiLayerNetwork, outputEvaluation : MultiLayerNetwork ) {
  def eval(feature : INDArray, neighborhood: List[NeighborhoodData]) : NodeState = {
    val state = neighborhood.map(neighbor => stateEvolution.output(neighbor.concatWithNodeFeature(feature), false))
      .foldRight[INDArray](new NDArray(feature.rows(), feature.columns()))((acc, data) => acc.addi(data))
    val result = outputEvaluation.output(state, false)
    NodeState(state, result)
  }
}
object GraphNeuralNetwork {
  case class NeighborhoodData(feature : INDArray, state : INDArray, edgeFeature : INDArray) {
    def concatWithNodeFeature(nodeFeature : INDArray) : INDArray = {
      new NDArray(nodeFeature :: feature :: state :: edgeFeature :: Nil map { node => node.toDoubleMatrix } reduce { _ ++ _ })
    }
  }
  case class NodeState(state : INDArray, output : INDArray)
}
