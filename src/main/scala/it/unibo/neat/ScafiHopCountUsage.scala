package it.unibo.neat

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import org.encog.ml.data.basic.BasicMLData
import org.encog.neural.neat.NEATNetwork

class ScafiHopCountUsage
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with FieldUtils {
  private val howMany = 10.0 //helps to normalize nodes..
  override def main(): Any = {
    val regression = node.get[NEATNetwork]("regression")
    rep(howMany) { data => {
      val elements = excludingSelf.reifyField(nbr(data)).values.toSeq :+ howMany
      val min = elements.min
      val normalized = min / howMany
      val sensor = if(node.has("source")) { 1.0 } else { 0.0 }
      val inputData = new BasicMLData(Array(normalized, sensor))
      val output = regression.compute(inputData).getData(0)
      node.put("output", output * howMany)
      output
    }}
  }
}
