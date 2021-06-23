package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.NeighborhoodData
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

class ScafiHopCountGNN extends AggregateProgram with FieldUtils with StandardSensors {
  private val network = sense[GraphNeuralNetwork]("network")
  private val initial = sense[Array[Float]]("initialState")
  private val initialFeature = sense[Array[Float]]("initialFeature")
  override def main(): Any = {
    rep[INDArray](new NDArray(initialFeature)) {
      feature => {
        rep[(INDArray, INDArray)]((new NDArray(Array(initial)), feature)) {
          case (state, output) =>
            val neighbourState = excludingSelf.reifyField(nbr(state))
            val labels = excludingSelf.reifyField(nbr(feature))
            val edgeLabels = excludingSelf.reifyField(nbr(new NDArray(Array(1.0f))))
            val neighborhoodData = neighbourState.keys.map(id => NeighborhoodData(labels(id), neighbourState(id), edgeLabels(id)))
              .toList
            val result = network.eval(feature, neighborhoodData)
            (result.state, result.output)
        }._2
      }
    }
  }
}
