---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# 📡 Kafka — Event-Driven Streaming
<!-- #endregion -->

<!-- #region -->
**Rôle visé** : Data Engineer · Wiki + Tutoriel

**Dataset(s)** : Synthetic event stream (générateur Python)

Kafka pour les architectures event-driven : pub/sub à haute volumétrie, exactly-once, schema evolution.

> Ce notebook est un **plan détaillé** des sections à aborder. Le code reste à implémenter — voir `00_critique.md` pour la priorisation.
<!-- #endregion -->

<!-- #region -->
## 1. Cadre messaging
<!-- #endregion -->

<!-- #region -->
Queues (1 consommateur) vs pub/sub (N consommateurs). Synchrone vs asynchrone. Push vs pull.
<!-- #endregion -->

<!-- #region -->
## 2. Architecture Kafka
<!-- #endregion -->

<!-- #region -->
Brokers, topics, partitions, replicas, ISR. Consumers in consumer groups. Offsets.
<!-- #endregion -->

<!-- #region -->
## 3. Producer / Consumer Python
<!-- #endregion -->

<!-- #region -->
`confluent-kafka-python` (perf maximale) ou `kafka-python` (pure Python). Code visé : producer continu + consumer manuel.
<!-- #endregion -->

<!-- #region -->
## 4. Schema Registry
<!-- #endregion -->

<!-- #region -->
Avro / Protobuf / JSON Schema. Évolution forward/backward compatible. Confluent Schema Registry.
<!-- #endregion -->

<!-- #region -->
## 5. Kafka Streams + ksqlDB
<!-- #endregion -->

<!-- #region -->
Stream processing dans la JVM. ksqlDB : SQL sur stream. Alternative : Flink.
<!-- #endregion -->

<!-- #region -->
## 6. Kafka Connect
<!-- #endregion -->

<!-- #region -->
Source connectors (DB → Kafka) et Sink connectors (Kafka → DB/S3). CDC avec Debezium.
<!-- #endregion -->

<!-- #region -->
## 7. Faust (Python streaming)
<!-- #endregion -->

<!-- #region -->
Frame Kafka Streams en Python. Async natif. Bon pour ML inference streaming.
<!-- #endregion -->

<!-- #region -->
## 8. Alternatives 2026
<!-- #endregion -->

<!-- #region -->
Apache Pulsar (multi-tenant), Redpanda (Kafka-compatible Rust), RabbitMQ (AMQP, plus queue que stream), NATS (pub/sub léger).
<!-- #endregion -->

<!-- #region -->
## 9. Event-driven architecture
<!-- #endregion -->

<!-- #region -->
CQRS, event sourcing, eventual consistency. Patterns Outbox / Saga.
<!-- #endregion -->

<!-- #region -->
## 10. Streaming ML
<!-- #endregion -->

<!-- #region -->
Online learning via `River`. Predictions streamées sur Kafka. Drift detection en stream.
<!-- #endregion -->

<!-- #region -->
## 11. Patterns
<!-- #endregion -->

<!-- #region -->
Exactly-once semantics (transactions Kafka), idempotent producers, deduplication, dead-letter queue.
<!-- #endregion -->

<!-- #region -->
## 12. Sources
<!-- #endregion -->

<!-- #region -->
- [Kafka docs](https://kafka.apache.org/documentation/)
- [Confluent Developer](https://developer.confluent.io/)
- [Faust docs](https://faust.readthedocs.io/)
- [Kafka: The Definitive Guide (livre Confluent free)](https://www.confluent.io/resources/kafka-the-definitive-guide/)
<!-- #endregion -->
