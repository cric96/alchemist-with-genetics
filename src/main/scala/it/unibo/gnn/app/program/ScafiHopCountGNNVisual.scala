package it.unibo.gnn.app.program

import io.jenetics.xml.Readers
import io.jenetics.{DoubleChromosome, DoubleGene, Genotype}
import it.unibo.gnn.app.{NetworkConfiguration, NetworkConfigurations}
import it.unibo.gnn.model.GraphNeuralNetwork
import it.unibo.gnn.model.GraphNeuralNetwork.NeighborhoodData
import it.unibo.gnn.model.NDArrayUtils.{nd, ndarray}
import it.unibo.scafi.incarnations.BasicSimulationIncarnation._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.io.FileInputStream
trait ScafiHopCountGNNVisual extends AggregateProgram with FieldUtils with StandardSensors {
  protected def config : NetworkConfiguration
  protected def genotype: Genotype[DoubleGene] = Readers.Genotype.read[java.lang.Double, DoubleGene, DoubleChromosome](
    new FileInputStream(config.outputFile),
    Readers.DoubleChromosome.reader()
  )
  protected val network : GraphNeuralNetwork = config.codec.loadFromGenotype(genotype)
  private val initialState =  (0 until config.stateSize).map(_ => -1f).toArray
  override def main(): Any = {
    val result = rep[Double](-1.0f) {
      out => {
        val (_, output) = rep[(INDArray, INDArray)]((ndarray(initialState), sourceFeature(out.toFloat))) {
          case (state, _) =>
            val neighbourState = excludingSelf.reifyField(nbr(state))
            val feature = sourceFeature(out.toFloat)
            val labels = excludingSelf.reifyField(feature)
            val edgeLabels = excludingSelf.reifyField(nbr(nd(1.0f)))
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

  def sourceFeature(out : Float) : INDArray = {
    val result = if(mid() == 0) { 1.0f } else { 0.0f }
    nd(result, out)
  }
}

class NonLinearGNNVisual extends ScafiHopCountGNNVisual {
  override protected def config: NetworkConfiguration = NetworkConfigurations.nonLinearConfig
}

class LinearGNNVisual extends ScafiHopCountGNNVisual {
  override protected def config: NetworkConfiguration = NetworkConfigurations.linearConfig
}

