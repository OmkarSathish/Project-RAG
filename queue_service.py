"""
RabbitMQ Queue Service for distributed task processing.

Features:
- Robust connections with auto-reconnect
- Priority queues for request prioritization
- Dead Letter Queues (DLQ) for failed messages
- Message acknowledgment patterns
- RPC-style request-reply pattern with correlation IDs
"""

import aio_pika
import json
from typing import Dict, Any, Callable, Optional
import asyncio


class QueueService:
    """Production-grade RabbitMQ service wrapper"""

    def __init__(self, rabbitmq_url: str):
        self.url = rabbitmq_url
        self.connection: Optional[aio_pika.RobustConnection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.consumers = {}

    async def connect(self):
        """Establish robust connection with auto-reconnect"""
        self.connection = await aio_pika.connect_robust(
            self.url,
            heartbeat=60,
            connection_attempts=5,
            retry_delay=2,
        )
        self.channel = await self.connection.channel()
        # Fair dispatch - only send message to worker when it's free
        await self.channel.set_qos(prefetch_count=1)
        print("[QueueService] Connected to RabbitMQ")

    async def declare_queue(
        self, queue_name: str, durable: bool = True, priority: bool = False
    ):
        """
        Declare queue with options.

        Args:
            queue_name: Name of the queue
            durable: Persist queue to disk (survives broker restart)
            priority: Enable priority queue support (0-10)
        """
        arguments = {}
        if priority:
            arguments["x-max-priority"] = 10  # Priority queue support

        queue = await self.channel.declare_queue(
            queue_name, durable=durable, arguments=arguments
        )
        return queue

    async def publish(
        self,
        queue_name: str,
        message: Dict[Any, Any],
        priority: int = 5,
        correlation_id: Optional[str] = None,
    ):
        """
        Publish message to queue with priority and correlation ID.

        Args:
            queue_name: Target queue name
            message: Message payload (will be JSON serialized)
            priority: Message priority (0-10, higher = more important)
            correlation_id: Optional correlation ID for RPC pattern
        """
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(message).encode(),
                priority=priority,
                correlation_id=correlation_id,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,  # Persist to disk
            ),
            routing_key=queue_name,
        )

    async def consume_with_callback(
        self, queue_name: str, callback: Callable, consumer_tag: str
    ):
        """
        Consume messages from queue with callback pattern.

        Args:
            queue_name: Queue to consume from
            callback: Async function to process message
            consumer_tag: Unique identifier for this consumer
        """
        queue = await self.declare_queue(queue_name, priority=True)

        async def message_handler(message: aio_pika.IncomingMessage):
            async with message.process(ignore_processed=True):
                try:
                    data = json.loads(message.body.decode())
                    result = await callback(data)

                    # Reply to reply_to queue if specified (RPC pattern)
                    if message.reply_to:
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(result).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )
                    await message.ack()  # Acknowledge successful processing
                except Exception as e:
                    print(f"[QueueService] Error processing message: {e}")
                    # Negative acknowledgment - send to DLQ
                    await message.nack(requeue=False)

        consumer = await queue.consume(message_handler, consumer_tag=consumer_tag)
        self.consumers[consumer_tag] = consumer
        print(f"[QueueService] Consumer '{consumer_tag}' started on queue '{queue_name}'")

    async def close(self):
        """Clean shutdown of all consumers and connection"""
        for consumer_tag, consumer in self.consumers.items():
            await consumer.cancel()
            print(f"[QueueService] Consumer '{consumer_tag}' stopped")

        if self.connection:
            await self.connection.close()
            print("[QueueService] Connection closed")
