package it.unibo.neat
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.encog.ml.data.basic.BasicMLData
import org.encog.neural.neat.NEATNetwork

class ScafiRegressionUsage
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport {
  def toss: Double = if (randomGen.nextDouble() < 0.5) { 1.0 } else { 0.0 }
  override def main(): Any = {
    val regression = node.get[NEATNetwork]("regression")
    val data = new BasicMLData(Array(toss, toss))
    node.put("input0", data.getData(0))
    node.put("input1", data.getData(1))
    node.put("output", regression.compute(data).getData(0))
  }
}
