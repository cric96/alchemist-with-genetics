package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.NeighborhoodData
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

class ScafiHopCountGNN extends AggregateProgram with FieldUtils with StandardSensors {
  override def main(): Any = {
    val network = sense[GraphNeuralNetwork]("network")
    val initial = sense[Array[Float]]("initialState")
    val result = rep[Double](-1.0f) {
      _ => {
        val (_, output) = rep[(INDArray, INDArray)]((new NDArray(Array(initial)), sourceFeature)) {
          case (state, _) =>
            val neighbourState = excludingSelf.reifyField(nbr(state))
            val feature = sourceFeature
            val labels = excludingSelf.reifyField(feature)
            val edgeLabels = excludingSelf.reifyField(nbr(new NDArray(Array(1.0f))))
            val neighborhoodData = neighbourState.keys.map(id => NeighborhoodData(labels(id), neighbourState(id), edgeLabels(id)))
              .toList
            val result = network.eval(feature, neighborhoodData)
            (result.state, result.output)
        }
        output.getDouble(0L)
      }
    }
    result
  }

  def sourceFeature : NDArray = new NDArray(Array(sense[Float]("source")))
}
