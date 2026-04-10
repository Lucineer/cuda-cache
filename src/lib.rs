/*!
# cuda-cache

Caching with TTL, LRU eviction, and namespaces.

Agents repeat themselves. Same query, same sensor reading, same
computation. Cache the result. This crate provides a generic
in-memory cache with LRU eviction, TTL expiration, and namespace
isolation so different subsystems don't collide.

- LRU (Least Recently Used) eviction
- TTL expiration per entry
- Namespaces for isolation
- Hit/miss statistics
- Cache warming (pre-populate)
- Capacity management
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// A cache entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: String,
    pub value: Vec<u8>,
    pub namespace: String,
    pub created_ms: u64,
    pub ttl_ms: Option<u64>,
    pub hits: u64,
    pub size_bytes: usize,
}

impl CacheEntry {
    pub fn is_expired(&self) -> bool {
        self.ttl_ms.map_or(false, |ttl| now() - self.created_ms > ttl)
    }
}

/// Cache statistics
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub total_sets: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { return 0.0; }
        self.hits as f64 / total as f64
    }
}

/// The cache
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cache {
    pub entries: HashMap<String, CacheEntry>,
    pub access_order: VecDeque<String>,  // for LRU
    pub max_size: usize,
    pub max_bytes: usize,
    pub current_bytes: usize,
    pub stats: CacheStats,
    pub default_ttl_ms: Option<u64>,
}

impl Cache {
    pub fn new(max_size: usize) -> Self { Cache { entries: HashMap::new(), access_order: VecDeque::new(), max_size, max_bytes: usize::MAX, current_bytes: 0, stats: CacheStats::default(), default_ttl_ms: None } }

    fn cache_key(&self, namespace: &str, key: &str) -> String { format!("{}:{}", namespace, key) }

    /// Get a value
    pub fn get(&mut self, namespace: &str, key: &str) -> Option<Vec<u8>> {
        let ck = self.cache_key(namespace, key);
        let entry = match self.entries.get_mut(&ck) {
            Some(e) if !e.is_expired() => e,
            _ => { self.stats.misses += 1; return None; }
        };
        entry.hits += 1;
        self.stats.hits += 1;
        // Move to front for LRU
        self.access_order.retain(|k| k != &ck);
        self.access_order.push_front(ck);
        Some(entry.value.clone())
    }

    /// Set a value
    pub fn set(&mut self, namespace: &str, key: &str, value: &[u8], ttl_ms: Option<u64>) {
        let ck = self.cache_key(namespace, key);
        // Remove existing if present
        if let Some(old) = self.entries.remove(&ck) { self.current_bytes -= old.size_bytes; }
        self.access_order.retain(|k| k != &ck);

        let ttl = ttl_ms.or(self.default_ttl_ms);
        let size = value.len();
        let entry = CacheEntry { key: key.to_string(), value: value.to_vec(), namespace: namespace.to_string(), created_ms: now(), ttl_ms: ttl, hits: 0, size_bytes: size };
        self.current_bytes += size;

        // Evict if needed
        while (self.entries.len() >= self.max_size || self.current_bytes > self.max_bytes) && !self.entries.is_empty() {
            self.evict_lru();
        }

        self.entries.insert(ck, entry);
        self.access_order.push_front(format!("{}:{}", namespace, key));
        self.stats.total_sets += 1;
    }

    /// Evict least recently used
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self.access_order.pop_back() {
            if let Some(entry) = self.entries.remove(&lru_key) {
                self.current_bytes -= entry.size_bytes;
                self.stats.evictions += 1;
            }
        }
    }

    /// Delete a key
    pub fn delete(&mut self, namespace: &str, key: &str) -> bool {
        let ck = self.cache_key(namespace, key);
        if let Some(entry) = self.entries.remove(&ck) {
            self.current_bytes -= entry.size_bytes;
            self.access_order.retain(|k| k != &ck);
            true
        } else { false }
    }

    /// Clear a namespace
    pub fn clear_namespace(&mut self, namespace: &str) -> usize {
        let prefix = format!("{}:", namespace);
        let to_remove: Vec<String> = self.entries.keys().filter(|k| k.starts_with(&prefix)).cloned().collect();
        for key in to_remove {
            if let Some(entry) = self.entries.remove(&key) { self.current_bytes -= entry.size_bytes; }
        }
        self.access_order.retain(|k| !k.starts_with(&prefix));
        to_remove.len()
    }

    /// Clean expired entries
    pub fn gc(&mut self) -> usize {
        let expired: Vec<String> = self.entries.iter().filter(|(_, e)| e.is_expired()).map(|(k, _)| k.clone()).collect();
        for key in &expired {
            if let Some(entry) = self.entries.remove(key) { self.current_bytes -= entry.size_bytes; self.stats.expirations += 1; }
            self.access_order.retain(|k| k != key);
        }
        expired.len()
    }

    /// Check if key exists
    pub fn has(&self, namespace: &str, key: &str) -> bool {
        let ck = self.cache_key(namespace, key);
        self.entries.get(&ck).map(|e| !e.is_expired()).unwrap_or(false)
    }

    /// Warm cache with multiple entries
    pub fn warm(&mut self, namespace: &str, entries: &[(&str, &[u8])]) {
        for (key, value) in entries { self.set(namespace, key, value, None); }
    }

    /// Get stats
    pub fn stats(&self) -> &CacheStats { &self.stats }

    /// Number of entries
    pub fn len(&self) -> usize { self.entries.len() }

    /// Summary
    pub fn summary(&self) -> String {
        format!("Cache: {}/{} entries ({:.0}% used), hit_rate={:.0}%, evictions={}, expirations={}",
            self.entries.len(), self.max_size,
            if self.max_size > 0 { self.entries.len() as f64 / self.max_size as f64 } else { 0.0 },
            self.stats.hit_rate() * 100.0, self.stats.evictions, self.stats.expirations)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut cache = Cache::new(100);
        cache.set("ns", "key1", b"value1", None);
        assert_eq!(cache.get("ns", "key1"), Some(b"value1".to_vec()));
    }

    #[test]
    fn test_miss() {
        let mut cache = Cache::new(100);
        assert_eq!(cache.get("ns", "missing"), None);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = Cache::new(3);
        cache.set("ns", "a", b"1", None);
        cache.set("ns", "b", b"2", None);
        cache.set("ns", "c", b"3", None);
        cache.get("ns", "a"); // touch a
        cache.set("ns", "d", b"4", None); // should evict b (least recent)
        assert!(cache.has("ns", "a")); // was touched
        assert!(!cache.has("ns", "b")); // evicted
    }

    #[test]
    fn test_ttl_expiration() {
        let mut cache = Cache::new(100);
        cache.set("ns", "temp", b"data", Some(0)); // immediate expiration
        assert_eq!(cache.get("ns", "temp"), None);
    }

    #[test]
    fn test_gc() {
        let mut cache = Cache::new(100);
        cache.set("ns", "expired", b"x", Some(0));
        cache.set("ns", "valid", b"y", Some(999_999_999));
        let removed = cache.gc();
        assert_eq!(removed, 1);
        assert!(cache.has("ns", "valid"));
    }

    #[test]
    fn test_namespace_isolation() {
        let mut cache = Cache::new(100);
        cache.set("ns1", "key", b"val1", None);
        cache.set("ns2", "key", b"val2", None);
        assert_eq!(cache.get("ns1", "key"), Some(b"val1".to_vec()));
        assert_eq!(cache.get("ns2", "key"), Some(b"val2".to_vec()));
    }

    #[test]
    fn test_clear_namespace() {
        let mut cache = Cache::new(100);
        cache.set("ns1", "a", b"1", None);
        cache.set("ns1", "b", b"2", None);
        cache.set("ns2", "c", b"3", None);
        let cleared = cache.clear_namespace("ns1");
        assert_eq!(cleared, 2);
        assert!(!cache.has("ns1", "a"));
        assert!(cache.has("ns2", "c"));
    }

    #[test]
    fn test_delete() {
        let mut cache = Cache::new(100);
        cache.set("ns", "key", b"val", None);
        assert!(cache.delete("ns", "key"));
        assert!(!cache.has("ns", "key"));
        assert!(!cache.delete("ns", "key")); // already gone
    }

    #[test]
    fn test_warm() {
        let mut cache = Cache::new(100);
        cache.warm("ns", &[("a", b"1"), ("b", b"2"), ("c", b"3")]);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = Cache::new(100);
        cache.set("ns", "key", b"val", None);
        cache.get("ns", "key"); // hit
        cache.get("ns", "miss"); // miss
        let rate = cache.stats().hit_rate();
        assert!((rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let cache = Cache::new(100);
        let s = cache.summary();
        assert!(s.contains("0/100"));
    }
}
