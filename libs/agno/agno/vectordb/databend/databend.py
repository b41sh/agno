from hashlib import md5
from typing import Any, Dict, List, Optional

from agno.vectordb.databend.index import HNSW

try:
    from databend_driver import BlockingDatabendClient
    from databend_driver import BlockingDatabendConnection
    from databend_driver import AsyncDatabendClient
    from databend_driver import AsyncDatabendConnection
except ImportError:
    raise ImportError("`databend-driver` not installed. Use `pip install databend-driver` to install it")

from agno.knowledge.document import Document
from agno.knowledge.embedder import Embedder
from agno.utils.log import log_debug, log_info, logger
from agno.vectordb.base import VectorDb
from agno.vectordb.distance import Distance

class Databend(VectorDb):
    """
    Databend class for managing vector operations with Databend.

    This class provides methods for creating, inserting, searching, and managing
    vector data in Databend.
    """

    def __init__(
        self,
        table_name: str,
        host: str,
        username: Optional[str] = None,
        password: str = "",
        port: int = 0,
        database_name: str = "ai",
        dsn: Optional[str] = None,
        compress: str = "lz4",
        client: Optional[BlockingDatabendConnection] = None,
        asyncclient: Optional[AsyncDatabendConnection] = None,
        embedder: Optional[Embedder] = None,
        distance: Distance = Distance.cosine,
        index: Optional[HNSW] = HNSW(),
    ):
        # Store connection parameters as instance attributes
        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.dsn = dsn
        self.compress = compress
        self.database_name = database_name

        if not client:
            databend_client = BlockingDatabendClient(
                f"databend://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}?sslmode=disable"
            )
            client = databend_client.get_conn()

        # Database attributes
        self.client = client
        self.async_client = asyncclient
        self.table_name = table_name

        # Embedder for embedding the document contents
        _embedder = embedder
        if _embedder is None:
            from agno.knowledge.embedder.openai import OpenAIEmbedder

            _embedder = OpenAIEmbedder()
            log_info("Embedder not provided, using OpenAIEmbedder as default.")
        self.embedder: Embedder = _embedder
        self.dimensions: Optional[int] = self.embedder.dimensions

        # Distance metric
        self.distance: Distance = distance

        # Index for the collection
        self.index: Optional[HNSW] = index

    async def _ensure_async_client(self):
        """Ensure we have an initialized async client."""
        if self.async_client is None:
            databend_async_client = AsyncDatabendClient(
                f"databend://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}?sslmode=disable"
            )
            self.async_client = await databend_async_client.get_conn()
        return self.async_client

    def _get_base_parameters(self) -> Dict[str, Any]:
        return {
            "table_name": self.table_name,
            "database_name": self.database_name,
        }

    def table_exists(self) -> bool:
        log_debug(f"Checking if table exists: {self.table_name}")
        try:
            result = self.client.query_row(
                f"EXISTS TABLE {self.database_name}.{self.table_name}",
            )
            if result is not None:
                return bool(result.values()[0])
            else:
                return False
        except Exception as e:
            logger.error(e)
            return False

    async def async_table_exists(self) -> bool:
        """Check if a table exists asynchronously."""
        log_debug(f"Async checking if table exists: {self.table_name}")
        try:
            async_client = await self._ensure_async_client()

            result = await async_client.query_row(
                "EXISTS TABLE {self.database_name}.{self.table_name}",
            )
            if result is not None:
                return bool(result.values()[0])
            else:
                return False
        except Exception as e:
            logger.error(f"Async error checking if table exists: {e}")
            return False

    def create(self) -> None:
        if not self.table_exists():
            log_debug(f"Creating Database: {self.database_name}")
            self.client.exec(
                f"CREATE DATABASE IF NOT EXISTS {self.database_name}"
            )

            log_debug(f"Creating table: {self.table_name}")

            parameters = self._get_base_parameters()

            if isinstance(self.index, HNSW):
                index = (
                    f"INDEX embedding_index embedding TYPE vector_similarity('hnsw', 'L2Distance', {self.embedder.dimensions}, {self.index.quantization}, "
                    f"{self.index.hnsw_max_connections_per_layer}, {self.index.hnsw_candidate_list_size_for_construction})"
                )
                #self.client.command("SET allow_experimental_vector_similarity_index = 1")
            else:
                raise NotImplementedError(f"Not implemented index {type(self.index)!r} is passed")

            self.client.exec(
                f"""CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name}
                (
                    id String,
                    name String,
                    meta_data Variant DEFAULT '{{}}',
                    filters Variant DEFAULT '{{}}',
                    content String,
                    content_id String,
                    embedding Array(Float32),
                    usage Variant,
                    created_at DateTime DEFAULT now(),
                    content_hash String
                ) ENGINE = Fuse""",
            )

    async def async_create(self) -> None:
        """Create database and table asynchronously."""
        if not await self.async_table_exists():
            log_debug(f"Async creating Database: {self.database_name}")
            async_client = await self._ensure_async_client()

            await async_client.exec(
                f"CREATE DATABASE IF NOT EXISTS {self.database_name}",
            )

            log_debug(f"Async creating table: {self.table_name}")

            if isinstance(self.index, HNSW):
                index = (
                    f"INDEX embedding_index embedding TYPE vector_similarity('hnsw', 'L2Distance', {self.index.quantization}, "
                    f"{self.index.hnsw_max_connections_per_layer}, {self.index.hnsw_candidate_list_size_for_construction})"
                )
                #await async_client.command("SET allow_experimental_vector_similarity_index = 1")
            else:
                raise NotImplementedError(f"Not implemented index {type(self.index)!r} is passed")

            #await self.async_client.command("SET enable_json_type = 1")  # type: ignore

            await self.async_client.exec(
                f"""CREATE TABLE IF NOT EXISTS {self.database_name}.{self.table_name}
                (
                    id String,
                    name String,
                    meta_data Variant DEFAULT '{{}}',
                    filters Variant DEFAULT '{{}}',
                    content String,
                    content_id String,
                    embedding Array(Float32),
                    usage variant,
                    created_at DateTime DEFAULT now(),
                    content_hash String,
                ) ENGINE = Fuse""",
            )

    def doc_exists(self, document: Document) -> bool:
        """
        Validating if the document exists or not

        Args:
            document (Document): Document to validate
        """
        cleaned_content = document.content.replace("\x00", "\ufffd")
        content_hash = md5(cleaned_content.encode()).hexdigest()

        result = self.client.query_row(
            f"SELECT content_hash FROM {self.database_name}.{self.table_name} WHERE content_hash = '{content_hash}'",
        )
        if result is not None:
            return bool(len(result) > 0)
        else:
            return False

    async def async_doc_exists(self, document: Document) -> bool:
        """Check if a document exists asynchronously."""
        async_client = await self._ensure_async_client()

        cleaned_content = document.content.replace("\x00", "\ufffd")
        content_hash = md5(cleaned_content.encode()).hexdigest()

        result = await async_client.query_row(
            f"SELECT content_hash FROM {self.database_name}.{self.table_name} WHERE content_hash = '{content_hash}'",
        )
        if result is not None:
            return bool(len(result) > 0)
        else:
            return False

    def name_exists(self, name: str) -> bool:
        """
        Validate if a row with this name exists or not

        Args:
            name (str): Name to check
        """

        result = self.client.query_row(
            f"SELECT name FROM {self.database_name}.{self.table_name} WHERE name = '{name}'",
        )
        if result is not None:
            return bool(len(result) > 0)
        else:
            return False

    async def async_name_exists(self, name: str) -> bool:
        """Check if a document with given name exists asynchronously."""
        async_client = await self._ensure_async_client()

        result = await async_client.query_row(
            f"SELECT name FROM {self.database_name}.{self.table_name} WHERE name = '{name}'",
        )
        if result is not None:
            return bool(len(result) > 0)
        else:
            return False

    def id_exists(self, id: str) -> bool:
        """
        Validate if a row with this id exists or not

        Args:
            id (str): Id to check
        """

        result = self.client.query_row(
            f"SELECT id FROM {self.database_name}.{self.table_name} WHERE id = '{id}'",
        )
        if result is not None:
            return bool(len(result) > 0)
        else:
            return False

    def insert(
        self,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        rows: List[List[Any]] = []
        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace("\x00", "\ufffd")
            content_hash = md5(cleaned_content.encode()).hexdigest()
            _id = document.id or content_hash

            row: List[Any] = [
                _id,
                document.name,
                document.meta_data,
                filters,
                cleaned_content,
                document.embedding,
                document.usage,
                content_hash,
            ]
            rows.append(row)

        self.client.insert(
            f"{self.database_name}.{self.table_name}",
            rows,
            column_names=[
                "id",
                "name",
                "meta_data",
                "filters",
                "content",
                "embedding",
                "usage",
                "content_hash",
            ],
        )
        log_debug(f"Inserted {len(documents)} documents")

    async def async_insert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Insert documents asynchronously."""
        rows: List[List[Any]] = []
        async_client = await self._ensure_async_client()

        for document in documents:
            document.embed(embedder=self.embedder)
            cleaned_content = document.content.replace("\x00", "\ufffd")
            content_hash = md5(cleaned_content.encode()).hexdigest()
            _id = document.id or content_hash

            row: List[Any] = [
                _id,
                document.name,
                document.meta_data,
                filters,
                cleaned_content,
                document.embedding,
                document.usage,
                content_hash,
            ]
            rows.append(row)

        await async_client.insert(
            f"{self.database_name}.{self.table_name}",
            rows,
            column_names=[
                "id",
                "name",
                "meta_data",
                "filters",
                "content",
                "embedding",
                "usage",
                "content_hash",
            ],
        )
        log_debug(f"Async inserted {len(documents)} documents")

    def upsert_available(self) -> bool:
        return True

    def upsert(
        self,
        documents: List[Document],
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert documents into the database.

        Args:
            documents (List[Document]): List of documents to upsert
            filters (Optional[Dict[str, Any]]): Filters to apply while upserting documents
            batch_size (int): Batch size for upserting documents
        """
        # We are using ReplacingMergeTree engine in our table, so we need to insert the documents,
        # then call SELECT with FINAL
        self.insert(documents=documents, filters=filters)

        parameters = self._get_base_parameters()
        self.client.query(
            "SELECT id FROM {database_name:Identifier}.{table_name:Identifier} FINAL",
            parameters=parameters,
        )

    async def async_upsert(self, documents: List[Document], filters: Optional[Dict[str, Any]] = None) -> None:
        """Upsert documents asynchronously."""
        # We are using ReplacingMergeTree engine in our table, so we need to insert the documents,
        # then call SELECT with FINAL
        await self.async_insert(documents=documents, filters=filters)

        parameters = self._get_base_parameters()
        await self.async_client.query(  # type: ignore
            "SELECT id FROM {database_name:Identifier}.{table_name:Identifier} FINAL",
            parameters=parameters,
        )

    def search(self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        parameters = self._get_base_parameters()
        where_query = ""
        # if filters:
        #     query_filters: List[str] = []
        #     for key, value in filters.values():
        #         query_filters.append(f"{{{key}_key:String}} = {{{key}_value:String}}")
        #         parameters[f"{key}_key"] = key
        #         parameters[f"{key}_value"] = value
        #     where_query = f"WHERE {' AND '.join(query_filters)}"

        order_by_query = ""
        if self.distance == Distance.l2 or self.distance == Distance.max_inner_product:
            order_by_query = "ORDER BY L2Distance(embedding, {query_embedding:Array(Float32)})"
            parameters["query_embedding"] = query_embedding
        if self.distance == Distance.cosine:
            order_by_query = "ORDER BY cosineDistance(embedding, {query_embedding:Array(Float32)})"
            parameters["query_embedding"] = query_embedding

        databend_query = (
            "SELECT name, meta_data, content, embedding, usage FROM "
            "{database_name:Identifier}.{table_name:Identifier} "
            f"{where_query} {order_by_query} LIMIT {limit}"
        )
        log_debug(f"Query: {databend_query}")
        log_debug(f"Params: {parameters}")

        try:
            results = self.client.query(
                databend_query,
                parameters=parameters,
            )
        except Exception as e:
            logger.error(f"Error searching for documents: {e}")
            logger.error("Table might not exist, creating for future use")
            self.create()
            return []

        # Build search results
        search_results: List[Document] = []
        for result in results.result_rows:
            search_results.append(
                Document(
                    name=result[0],
                    meta_data=result[1],
                    content=result[2],
                    embedder=self.embedder,
                    embedding=result[3],
                    usage=result[4],
                )
            )

        return search_results

    async def async_search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for documents asynchronously."""
        async_client = await self._ensure_async_client()

        query_embedding = self.embedder.get_embedding(query)
        if query_embedding is None:
            logger.error(f"Error getting embedding for Query: {query}")
            return []

        parameters = self._get_base_parameters()
        where_query = ""
        # if filters:
        #     query_filters: List[str] = []
        #     for key, value in filters.values():
        #         query_filters.append(f"{{{key}_key:String}} = {{{key}_value:String}}")
        #         parameters[f"{key}_key"] = key
        #         parameters[f"{key}_value"] = value
        #     where_query = f"WHERE {' AND '.join(query_filters)}"

        order_by_query = ""
        if self.distance == Distance.l2 or self.distance == Distance.max_inner_product:
            order_by_query = "ORDER BY L2Distance(embedding, {query_embedding:Array(Float32)})"
            parameters["query_embedding"] = query_embedding
        if self.distance == Distance.cosine:
            order_by_query = "ORDER BY cosineDistance(embedding, {query_embedding:Array(Float32)})"
            parameters["query_embedding"] = query_embedding

        databend_query = (
            "SELECT name, meta_data, content, embedding, usage FROM "
            "{database_name:Identifier}.{table_name:Identifier} "
            f"{where_query} {order_by_query} LIMIT {limit}"
        )
        log_debug(f"Async Query: {databend_query}")
        log_debug(f"Async Params: {parameters}")

        try:
            results = await async_client.query(
                databend_query,
                parameters=parameters,
            )
        except Exception as e:
            logger.error(f"Async error searching for documents: {e}")
            logger.error("Table might not exist, creating for future use")
            await self.async_create()
            return []

        # Build search results
        search_results: List[Document] = []
        for result in results.result_rows:
            search_results.append(
                Document(
                    name=result[0],
                    meta_data=result[1],
                    content=result[2],
                    embedder=self.embedder,
                    embedding=result[3],
                    usage=result[4],
                )
            )

        return search_results

    def drop(self) -> None:
        if self.table_exists():
            log_debug(f"Deleting table: {self.table_name}")
            self.client.exec(
                f"DROP TABLE {self.database_name}.{self.table_name}",
            )

    async def async_drop(self) -> None:
        """Drop the table asynchronously."""
        if await self.async_exists():
            log_debug(f"Async dropping table: {self.table_name}")
            await self.async_client.exec(
                f"DROP TABLE {self.database_name}.{self.table_name}",
            )

    def exists(self) -> bool:
        return self.table_exists()

    async def async_exists(self) -> bool:
        return await self.async_table_exists()

    def get_count(self) -> int:
        result = self.client.query_row(
            f"SELECT count(*) FROM {self.database_name}.{self.table_name}",
        )

        if result is not None:
            return int(result.values()[0])
        return 0

    def optimize(self) -> None:
        log_debug("==== No need to optimize Databend. Skipping this step ====")

    def delete(self) -> bool:
        self.client.exec(
            f"DELETE FROM {self.database_name}.{self.table_name}",
        )
        return True

    def delete_by_id(self, id: str) -> bool:
        """

        Delete a document by its ID.

        Args:
            id (str): The document ID to delete

        Returns:
            bool: True if document was deleted, False otherwise
        """
        try:
            log_debug(f"Databend : Deleting document with ID {id}")
            if not self.id_exists(id):
                return False

            self.client.exec(
                f"DELETE FROM {self.database_name}.{self.table_name} WHERE id = '{id}'",
            )
            return True
        except Exception as e:
            log_info(f"Error deleting document with ID {id}: {e}")
            return False

    def delete_by_name(self, name: str) -> bool:
        """
        Delete documents by name.

        Args:
            name (str): The document name to delete

        Returns:
            bool: True if documents were deleted, False otherwise
        """
        try:
            log_debug(f"Databend : Deleting documents with name {name}")
            if not self.name_exists(name):
                return False

            self.client.exec(
                f"DELETE FROM {self.database_name}.{self.table_name} WHERE name = '{name}'",
            )
            return True
        except Exception as e:
            log_info(f"Error deleting documents with name {name}: {e}")
            return False

    def delete_by_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Delete documents by metadata.

        Args:
            metadata (Dict[str, Any]): The metadata to match for deletion

        Returns:
            bool: True if documents were deleted, False otherwise
        """
        try:
            log_debug(f"Databend : Deleting documents with metadata {metadata}")

            # Build WHERE clause for metadata matching using proper ClickHouse JSON syntax
            where_conditions = []
            for key, value in metadata.items():
                if isinstance(value, bool):
                    where_conditions.append(f"filters['{key}'] = {str(value).lower()}")
                elif isinstance(value, (int, float)):
                    where_conditions.append(f"filters['{key}'] = {value}")
                else:
                    where_conditions.append(f"filters['{key}'] = '{value}'")

            if not where_conditions:
                return False

            where_clause = " AND ".join(where_conditions)

            self.client.exec(
                f"DELETE FROM {self.database_name}.{self.table_name} WHERE {where_clause}",
            )
            return True
        except Exception as e:
            log_info(f"Error deleting documents with metadata {metadata}: {e}")
            return False

    def delete_by_content_id(self, content_id: str) -> bool:
        """
        Delete documents by content ID.

        Args:
            content_id (str): The content ID to delete

        Returns:
            bool: True if documents were deleted, False otherwise
        """
        try:
            log_debug(f"Databend : Deleting documents with content_id {content_id}")

            self.client.exec(
                f"DELETE FROM {self.database_name}.{self.table_name} WHERE content_id = '{content_id}'",
            )
            return True
        except Exception as e:
            log_info(f"Error deleting documents with content_id {content_id}: {e}")
            return False

    def content_hash_exists(self, content_hash: str) -> bool:
        """
        Validate if a row with this content_hash exists or not

        Args:
            content_hash (str): Content hash to check
        """
        result = self.client.query_row(
            f"SELECT content_hash FROM {self.database_name}.{self.table_name} WHERE content_hash = '{content_hash}'",
        )
        if not result or len(result) == 0:
            return False
        else:
            return True

    def _delete_by_content_hash(self, content_hash: str) -> bool:
        """
        Delete documents by content hash.
        """
        try:
            self.client.exec(
                f"DELETE FROM {self.database_name}.{self.table_name} WHERE content_hash = '{content_hash}'",
            )
            return True
        except Exception:
            return False

    def update_metadata(self, content_id: str, metadata: Dict[str, Any]) -> None:
        """
        Update the metadata for documents with the given content_id.

        Args:
            content_id (str): The content ID to update
            metadata (Dict[str, Any]): The metadata to update
        """
        import json

        try:
            # First, get existing documents with their current metadata and filters
            result = self.client.query_all(
                f"SELECT id, meta_data, filters FROM {self.database_name}.{self.table_name} WHERE content_id = '{content_id}'",
            )

            if result is None or len(result) == 0:
                logger.debug(f"No documents found with content_id: {content_id}")
                return

            # Update each document
            updated_count = 0
            for row in result:
                doc_id, current_meta_json, current_filters_json = row.values()

                # Parse existing metadata
                try:
                    current_metadata = json.loads(current_meta_json) if current_meta_json else {}
                except (json.JSONDecodeError, TypeError):
                    current_metadata = {}

                # Parse existing filters
                try:
                    current_filters = json.loads(current_filters_json) if current_filters_json else {}
                except (json.JSONDecodeError, TypeError):
                    current_filters = {}

                # Merge existing metadata with new metadata
                updated_metadata = current_metadata.copy()
                updated_metadata.update(metadata)

                # Merge existing filters with new metadata
                updated_filters = current_filters.copy()
                updated_filters.update(metadata)

                # Update the document
                update_params = parameters.copy()
                metadata_json = json.dumps(updated_metadata)
                filters_json = json.dumps(updated_filters)

                self.client.command(
                    f"ALTER TABLE {self.database_name}.{self.table_name} UPDATE meta_data = '{metadata_json}', filters = '{filters_json}' WHERE id = '{doc_id}'",
                )
                updated_count += 1

            logger.debug(f"Updated metadata for {updated_count} documents with content_id: {content_id}")

        except Exception as e:
            logger.error(f"Error updating metadata for content_id '{content_id}': {e}")
            raise





