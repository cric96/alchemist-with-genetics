package it.unibo.gnn.app.program

import io.jenetics.xml.Readers
import io.jenetics.{DoubleChromosome, DoubleGene}
import it.unibo.gnn.app.NetworkConfiguration
import it.unibo.gnn.model.GraphNeuralNetwork.NeighborhoodData
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.cpu.nativecpu.NDArray

import java.io.FileInputStream

class ScafiHopCountGNNVisual extends AggregateProgram with FieldUtils with StandardSensors {
  private val genotype = Readers.Genotype.read[java.lang.Double, DoubleGene, DoubleChromosome](new FileInputStream("result.xml"), Readers.DoubleChromosome.reader())
  private val network = NetworkConfiguration.nonLinearCodec.loadFromGenotype(genotype)
  private val initialState = Array[Float](-1f, -1f, -1f, -1f)
  override def main(): Any = {
    val result = rep[Double](-1.0f) {
      _ => {
        val (_, output) = rep[(INDArray, INDArray)]((new NDArray(initialState), sourceFeature)) {
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

  def sourceFeature : NDArray = {
    val result = if(mid() == 0) { 1.0f } else { 0.0f }
    new NDArray(Array(Array(result)))
  }
}
