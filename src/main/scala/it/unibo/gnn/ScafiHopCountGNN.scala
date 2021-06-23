package it.unibo.gnn

import it.unibo.gnn.GraphNeuralNetwork.NeighborhoodData
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

class ScafiHopCountGNN extends AggregateProgram with FieldUtils with StandardSensors {
  override def main(): Any = {
    val network = sense[GraphNeuralNetwork]("network")
    val initial = sense[Array[Float]]("initialState")
    val innerLoop = rep[INDArray](new NDArray(Array(-1.0f, sourceFeature))) {
      feature => {
        val (_, output) = rep[(INDArray, INDArray)]((new NDArray(Array(initial)), feature)) {
          case (state, _) =>
            val neighbourState = excludingSelf.reifyField(nbr(state))
            val labels = excludingSelf.reifyField(nbr(feature))
            val edgeLabels = excludingSelf.reifyField(nbr(new NDArray(Array(1.0f))))
            val neighborhoodData = neighbourState.keys.map(id => NeighborhoodData(labels(id), neighbourState(id), edgeLabels(id)))
              .toList
            val result = network.eval(feature, neighborhoodData)
            (result.state, result.output)
        }
        new NDArray(Array(output.getDouble(0L).toFloat, sourceFeature))
      }
    }
    innerLoop.getDouble(0L)
  }

  def sourceFeature : Float = sense("source")
}
