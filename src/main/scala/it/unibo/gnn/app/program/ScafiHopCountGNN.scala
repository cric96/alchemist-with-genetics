package it.unibo.gnn.app.program

import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.gnn.model.GraphNeuralNetwork.NeighborhoodData
import it.unibo.gnn.model.NDArrayUtils.{nd, ndarray}
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

class ScafiHopCountGNN extends AggregateProgram with FieldUtils with StandardSensors {
  private val constantField = Array(1.0f)
  override def main(): Any = {
    val network = sense[GraphNeuralNetwork]("network")
    val initial = sense[Array[Float]]("initialState")
    val result = rep[Double](-1.0f) {
      out => {
        val (_, output) = rep[(INDArray, INDArray)]((ndarray((initial)), sourceFeature(out.toFloat))) {
          case (state, _) =>
            val neighbourState = excludingSelf.reifyField(nbr(state))
            val feature = sourceFeature(out.toFloat)
            val labels = excludingSelf.reifyField(feature)
            val edgeLabels = excludingSelf.reifyField(nbr(ndarray(constantField)))
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

  def sourceFeature(output : Float) : INDArray = nd(sense[Float]("source"), output)
}
