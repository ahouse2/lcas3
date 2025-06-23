"""
Timeline Agent
Specialized in temporal analysis, event sequencing, and chronological pattern discovery
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from .base_agent import BaseAgent, AgentResult

class TimelineAgent(BaseAgent):
    """Agent specialized in timeline analysis and temporal pattern discovery"""
    
    def __init__(self, ai_service=None, config: Dict[str, Any] = None):
        super().__init__("Timeline", ai_service, config)
        
        # Date pattern configurations
        self.date_patterns = [
            {
                'pattern': r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
                'format': 'MM/DD/YYYY',
                'confidence': 0.9
            },
            {
                'pattern': r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', 
                'format': 'MM-DD-YYYY',
                'confidence': 0.9
            },
            {
                'pattern': r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
                'format': 'YYYY-MM-DD',
                'confidence': 0.95
            },
            {
                'pattern': r'\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b',
                'format': 'Month DD, YYYY',
                'confidence': 0.85
            }
        ]
        
        # Time pattern configurations
        self.time_patterns = [
            r'\b(\d{1,2}):(\d{2})\s*([AP]M)\b',
            r'\b(\d{1,2}):(\d{2}):(\d{2})\b',
            r'\b(\d{1,2}):(\d{2})\b'
        ]
        
    def get_capabilities(self) -> List[str]:
        return [
            "temporal_event_extraction",
            "chronological_sequencing", 
            "timeline_pattern_analysis",
            "date_time_normalization",
            "temporal_relationship_mapping",
            "event_clustering",
            "timeline_gap_analysis"
        ]
    
    async def analyze(self, data: Any, context: Dict[str, Any] = None) -> AgentResult:
        """Analyze document for timeline events and temporal patterns"""
        start_time = datetime.now()
        
        if not await self.validate_input(data):
            return self._create_error_result("Invalid input data", start_time)
        
        try:
            content = data.get('content', '')
            file_path = data.get('file_path', '')
            
            # Extract temporal events
            temporal_events = await self._extract_temporal_events(content, file_path)
            
            # Analyze temporal patterns
            pattern_analysis = await self._analyze_temporal_patterns(temporal_events)
            
            # Sequence events chronologically
            chronological_sequence = await self._create_chronological_sequence(temporal_events)
            
            # Identify temporal relationships
            relationship_analysis = await self._analyze_temporal_relationships(temporal_events)
            
            # Detect timeline gaps and anomalies
            gap_analysis = await self._analyze_timeline_gaps(temporal_events)
            
            # AI-enhanced timeline analysis
            ai_analysis = {}
            if self.ai_service:
                ai_analysis = await self._ai_timeline_analysis(content, temporal_events, context)
            
            findings = {
                "temporal_events": temporal_events,
                "pattern_analysis": pattern_analysis,
                "chronological_sequence": chronological_sequence,
                "relationship_analysis": relationship_analysis,
                "gap_analysis": gap_analysis,
                "ai_analysis": ai_analysis
            }
            
            confidence = self.calculate_confidence(findings)
            evidence_strength = self._calculate_temporal_strength(findings)
            legal_significance = self.extract_legal_significance(findings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AgentResult(
                agent_name=self.name,
                analysis_type="timeline_analysis",
                confidence=confidence,
                findings=findings,
                recommendations=self._generate_recommendations(findings),
                evidence_strength=evidence_strength,
                legal_significance=legal_significance,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={"file_path": file_path, "events_found": len(temporal_events)}
            )
            
        except Exception as e:
            self.logger.error(f"Timeline analysis failed: {e}")
            return self._create_error_result(str(e), start_time)
    
    async def _extract_temporal_events(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract temporal events from content"""
        events = []
        
        # Extract date mentions with context
        date_mentions = self._extract_dates_with_context(content)
        
        for date_info in date_mentions:
            # Extract surrounding context for event description
            context_start = max(0, date_info['position'] - 150)
            context_end = min(len(content), date_info['position'] + 150)
            event_context = content[context_start:context_end].strip()
            
            # Classify event type based on context
            event_type = self._classify_event_type(event_context)
            
            # Extract participants/entities
            participants = self._extract_event_participants(event_context)
            
            # Assess event significance
            significance = self._assess_event_significance(event_context, event_type)
            
            event = {
                "date": date_info['parsed_date'],
                "date_confidence": date_info['confidence'],
                "raw_date_text": date_info['raw_date'],
                "event_description": self._extract_event_description(event_context),
                "event_type": event_type,
                "participants": participants,
                "significance_score": significance,
                "context": event_context,
                "source_file": file_path,
                "position_in_document": date_info['position']
            }
            
            # Extract time if present
            time_info = self._extract_time_from_context(event_context)
            if time_info:
                event.update(time_info)
            
            events.append(event)
        
        return events
    
    def _extract_dates_with_context(self, content: str) -> List[Dict[str, Any]]:
        """Extract dates with their context from content"""
        date_mentions = []
        
        for pattern_info in self.date_patterns:
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                try:
                    date_str = match.group()
                    parsed_date = self._parse_date(date_str, pattern_info['format'])
                    
                    if parsed_date:
                        date_mentions.append({
                            'raw_date': date_str,
                            'parsed_date': parsed_date,
                            'confidence': pattern_info['confidence'],
                            'position': match.start(),
                            'format': pattern_info['format']
                        })
                        
                except Exception as e:
                    self.logger.debug(f"Error parsing date {match.group()}: {e}")
                    continue
        
        # Remove duplicates and sort by position
        unique_dates = []
        seen_positions = set()
        
        for date_info in sorted(date_mentions, key=lambda x: x['position']):
            # Skip if too close to a previous date (likely duplicate)
            if not any(abs(date_info['position'] - pos) < 10 for pos in seen_positions):
                unique_dates.append(date_info)
                seen_positions.add(date_info['position'])
        
        return unique_dates
    
    def _parse_date(self, date_str: str, format_hint: str) -> Optional[str]:
        """Parse date string into ISO format"""
        try:
            if format_hint == 'MM/DD/YYYY':
                dt = datetime.strptime(date_str, '%m/%d/%Y')
            elif format_hint == 'MM-DD-YYYY':
                dt = datetime.strptime(date_str, '%m-%d-%Y')
            elif format_hint == 'YYYY-MM-DD':
                dt = datetime.strptime(date_str, '%Y-%m-%d')
            elif format_hint == 'Month DD, YYYY':
                # Try different month formats
                for fmt in ['%B %d, %Y', '%b %d, %Y', '%B %d %Y', '%b %d %Y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
            else:
                return None
            
            return dt.strftime('%Y-%m-%d')
            
        except ValueError:
            return None
    
    def _extract_time_from_context(self, context: str) -> Optional[Dict[str, Any]]:
        """Extract time information from context"""
        for pattern in self.time_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return {
                    "time": match.group(),
                    "time_confidence": 0.8,
                    "has_time": True
                }
        
        return {"has_time": False}
    
    def _classify_event_type(self, context: str) -> str:
        """Classify the type of event based on context"""
        context_lower = context.lower()
        
        # Event type classifications
        event_types = {
            "communication": ["email", "text", "call", "message", "sent", "received", "phone"],
            "meeting": ["meeting", "appointment", "conference", "visit", "saw", "met"],
            "financial": ["payment", "deposit", "withdrawal", "transfer", "purchase", "sale", "$"],
            "legal": ["filed", "served", "court", "hearing", "motion", "order", "judgment"],
            "medical": ["doctor", "hospital", "treatment", "appointment", "surgery", "diagnosis"],
            "travel": ["flight", "trip", "travel", "arrived", "departed", "vacation"],
            "work": ["work", "job", "office", "business", "employment", "meeting"],
            "personal": ["birthday", "anniversary", "wedding", "funeral", "graduation"],
            "incident": ["accident", "injury", "incident", "problem", "issue", "emergency"]
        }
        
        # Score each type
        type_scores = {}
        for event_type, keywords in event_types.items():
            score = sum(1 for keyword in keywords if keyword in context_lower)
            if score > 0:
                type_scores[event_type] = score
        
        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        else:
            return "general"
    
    def _extract_event_participants(self, context: str) -> List[str]:
        """Extract participants/entities from event context"""
        participants = []
        
        # Extract proper names (simple pattern)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        names = re.findall(name_pattern, context)
        
        # Filter out common false positives
        false_positives = {"United States", "New York", "Los Angeles", "San Francisco", "Supreme Court"}
        names = [name for name in names if name not in false_positives]
        
        participants.extend(names)
        
        # Extract pronouns and roles
        role_patterns = [
            r'\b(?:plaintiff|defendant|petitioner|respondent|witness|attorney|judge|doctor|officer)\b'
        ]
        
        for pattern in role_patterns:
            roles = re.findall(pattern, context, re.IGNORECASE)
            participants.extend(roles)
        
        return list(set(participants))  # Remove duplicates
    
    def _assess_event_significance(self, context: str, event_type: str) -> float:
        """Assess the significance of an event"""
        significance = 0.5  # Base significance
        
        # Type-based significance
        type_significance = {
            "legal": 0.9,
            "financial": 0.8,
            "incident": 0.8,
            "medical": 0.7,
            "communication": 0.6,
            "meeting": 0.5,
            "work": 0.4,
            "travel": 0.3,
            "personal": 0.3,
            "general": 0.2
        }
        
        significance = type_significance.get(event_type, 0.5)
        
        # Context-based adjustments
        context_lower = context.lower()
        
        # High significance indicators
        high_sig_indicators = ["court", "lawsuit", "fraud", "breach", "violation", "emergency", "arrest"]
        if any(indicator in context_lower for indicator in high_sig_indicators):
            significance += 0.2
        
        # Medium significance indicators
        med_sig_indicators = ["contract", "agreement", "payment", "meeting", "appointment"]
        if any(indicator in context_lower for indicator in med_sig_indicators):
            significance += 0.1
        
        # Specific amounts or precise details increase significance
        if re.search(r'\$[\d,]+', context) or re.search(r'\d{1,2}:\d{2}', context):
            significance += 0.1
        
        return min(significance, 1.0)
    
    def _extract_event_description(self, context: str) -> str:
        """Extract a concise event description from context"""
        # Find the sentence containing the most relevant information
        sentences = re.split(r'[.!?]+', context)
        
        if not sentences:
            return context[:100] + "..." if len(context) > 100 else context
        
        # Score sentences based on content
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            score = 0
            
            # Prefer sentences with action words
            action_words = ["filed", "sent", "received", "met", "called", "paid", "signed", "agreed"]
            score += sum(1 for word in action_words if word in sentence.lower())
            
            # Prefer sentences with specific details
            if re.search(r'\$[\d,]+', sentence):
                score += 2
            if re.search(r'\d{1,2}:\d{2}', sentence):
                score += 1
            
            # Prefer sentences with proper names
            if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', sentence):
                score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        if best_sentence:
            return best_sentence[:150] + "..." if len(best_sentence) > 150 else best_sentence
        else:
            return sentences[0][:150] + "..." if len(sentences[0]) > 150 else sentences[0]
    
    async def _analyze_temporal_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in temporal events"""
        if not events:
            return {"patterns": [], "pattern_count": 0}
        
        patterns = []
        
        # Sort events by date
        sorted_events = sorted([e for e in events if e.get('date')], 
                             key=lambda x: x['date'])
        
        # Analyze event frequency patterns
        frequency_patterns = self._analyze_event_frequency(sorted_events)
        patterns.extend(frequency_patterns)
        
        # Analyze event type clustering
        clustering_patterns = self._analyze_event_clustering(sorted_events)
        patterns.extend(clustering_patterns)
        
        # Analyze temporal gaps
        gap_patterns = self._analyze_temporal_gaps(sorted_events)
        patterns.extend(gap_patterns)
        
        # Analyze escalation patterns
        escalation_patterns = self._analyze_escalation_patterns(sorted_events)
        patterns.extend(escalation_patterns)
        
        return {
            "patterns": patterns,
            "pattern_count": len(patterns),
            "total_events": len(events),
            "date_range": {
                "earliest": sorted_events[0]['date'] if sorted_events else None,
                "latest": sorted_events[-1]['date'] if sorted_events else None
            }
        }
    
    def _analyze_event_frequency(self, sorted_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze frequency patterns in events"""
        patterns = []
        
        if len(sorted_events) < 3:
            return patterns
        
        # Group events by month
        monthly_counts = {}
        for event in sorted_events:
            date_str = event['date']
            month_key = date_str[:7]  # YYYY-MM
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        
        # Find high-activity periods
        avg_monthly = sum(monthly_counts.values()) / len(monthly_counts)
        high_activity_months = {month: count for month, count in monthly_counts.items() 
                              if count > avg_monthly * 1.5}
        
        if high_activity_months:
            patterns.append({
                "type": "high_frequency_periods",
                "description": f"High activity periods identified: {len(high_activity_months)} months",
                "details": high_activity_months,
                "significance": 0.7
            })
        
        return patterns
    
    def _analyze_event_clustering(self, sorted_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze clustering of similar event types"""
        patterns = []
        
        # Group events by type
        type_groups = {}
        for event in sorted_events:
            event_type = event.get('event_type', 'general')
            if event_type not in type_groups:
                type_groups[event_type] = []
            type_groups[event_type].append(event)
        
        # Analyze clustering within each type
        for event_type, events in type_groups.items():
            if len(events) >= 3:
                # Check for temporal clustering
                clusters = self._find_temporal_clusters(events)
                if clusters:
                    patterns.append({
                        "type": "event_type_clustering",
                        "description": f"{event_type} events show clustering pattern",
                        "event_type": event_type,
                        "clusters": clusters,
                        "significance": 0.6
                    })
        
        return patterns
    
    def _find_temporal_clusters(self, events: List[Dict[str, Any]], max_gap_days: int = 7) -> List[Dict[str, Any]]:
        """Find temporal clusters in events"""
        if len(events) < 2:
            return []
        
        sorted_events = sorted(events, key=lambda x: x['date'])
        clusters = []
        current_cluster = [sorted_events[0]]
        
        for i in range(1, len(sorted_events)):
            prev_date = datetime.strptime(sorted_events[i-1]['date'], '%Y-%m-%d')
            curr_date = datetime.strptime(sorted_events[i]['date'], '%Y-%m-%d')
            
            if (curr_date - prev_date).days <= max_gap_days:
                current_cluster.append(sorted_events[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append({
                        "start_date": current_cluster[0]['date'],
                        "end_date": current_cluster[-1]['date'],
                        "event_count": len(current_cluster),
                        "events": current_cluster
                    })
                current_cluster = [sorted_events[i]]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append({
                "start_date": current_cluster[0]['date'],
                "end_date": current_cluster[-1]['date'],
                "event_count": len(current_cluster),
                "events": current_cluster
            })
        
        return clusters
    
    def _analyze_temporal_gaps(self, sorted_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze significant gaps in timeline"""
        patterns = []
        
        if len(sorted_events) < 2:
            return patterns
        
        gaps = []
        for i in range(1, len(sorted_events)):
            prev_date = datetime.strptime(sorted_events[i-1]['date'], '%Y-%m-%d')
            curr_date = datetime.strptime(sorted_events[i]['date'], '%Y-%m-%d')
            gap_days = (curr_date - prev_date).days
            
            if gap_days > 30:  # Significant gap
                gaps.append({
                    "start_date": sorted_events[i-1]['date'],
                    "end_date": sorted_events[i]['date'],
                    "gap_days": gap_days,
                    "before_event": sorted_events[i-1]['event_description'],
                    "after_event": sorted_events[i]['event_description']
                })
        
        if gaps:
            patterns.append({
                "type": "significant_gaps",
                "description": f"Found {len(gaps)} significant timeline gaps",
                "gaps": gaps,
                "significance": 0.5
            })
        
        return patterns
    
    def _analyze_escalation_patterns(self, sorted_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze escalation patterns in events"""
        patterns = []
        
        if len(sorted_events) < 3:
            return patterns
        
        # Look for increasing significance over time
        significance_trend = [event.get('significance_score', 0.5) for event in sorted_events]
        
        # Simple trend analysis
        increasing_periods = 0
        for i in range(1, len(significance_trend)):
            if significance_trend[i] > significance_trend[i-1]:
                increasing_periods += 1
        
        escalation_ratio = increasing_periods / (len(significance_trend) - 1)
        
        if escalation_ratio > 0.6:  # More than 60% increasing
            patterns.append({
                "type": "escalation_pattern",
                "description": "Events show escalating significance over time",
                "escalation_ratio": escalation_ratio,
                "significance": 0.8
            })
        
        return patterns
    
    async def _create_chronological_sequence(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chronological sequence of events"""
        # Filter events with valid dates and sort
        valid_events = [e for e in events if e.get('date')]
        sorted_events = sorted(valid_events, key=lambda x: x['date'])
        
        # Add sequence information
        for i, event in enumerate(sorted_events):
            event['sequence_number'] = i + 1
            event['is_first'] = (i == 0)
            event['is_last'] = (i == len(sorted_events) - 1)
            
            # Calculate time since previous event
            if i > 0:
                prev_date = datetime.strptime(sorted_events[i-1]['date'], '%Y-%m-%d')
                curr_date = datetime.strptime(event['date'], '%Y-%m-%d')
                event['days_since_previous'] = (curr_date - prev_date).days
            else:
                event['days_since_previous'] = 0
        
        return sorted_events
    
    async def _analyze_temporal_relationships(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships between temporal events"""
        relationships = {
            "causal_relationships": [],
            "concurrent_events": [],
            "sequential_patterns": []
        }
        
        if len(events) < 2:
            return relationships
        
        sorted_events = sorted([e for e in events if e.get('date')], 
                             key=lambda x: x['date'])
        
        # Find concurrent events (same date)
        date_groups = {}
        for event in sorted_events:
            date = event['date']
            if date not in date_groups:
                date_groups[date] = []
            date_groups[date].append(event)
        
        for date, events_on_date in date_groups.items():
            if len(events_on_date) > 1:
                relationships["concurrent_events"].append({
                    "date": date,
                    "events": events_on_date,
                    "event_count": len(events_on_date)
                })
        
        # Analyze potential causal relationships
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Check if events might be causally related
            causal_score = self._assess_causal_relationship(current_event, next_event)
            if causal_score > 0.5:
                relationships["causal_relationships"].append({
                    "cause_event": current_event,
                    "effect_event": next_event,
                    "causal_score": causal_score,
                    "time_gap_days": next_event.get('days_since_previous', 0)
                })
        
        return relationships
    
    def _assess_causal_relationship(self, event1: Dict[str, Any], event2: Dict[str, Any]) -> float:
        """Assess potential causal relationship between two events"""
        score = 0.0
        
        # Time proximity increases causal likelihood
        time_gap = event2.get('days_since_previous', 999)
        if time_gap <= 1:
            score += 0.4
        elif time_gap <= 7:
            score += 0.2
        elif time_gap <= 30:
            score += 0.1
        
        # Event type relationships
        type1 = event1.get('event_type', '')
        type2 = event2.get('event_type', '')
        
        causal_type_pairs = {
            ("communication", "meeting"): 0.3,
            ("meeting", "legal"): 0.4,
            ("incident", "medical"): 0.5,
            ("incident", "legal"): 0.4,
            ("financial", "legal"): 0.3
        }
        
        score += causal_type_pairs.get((type1, type2), 0.0)
        
        # Content-based causal indicators
        desc1 = event1.get('event_description', '').lower()
        desc2 = event2.get('event_description', '').lower()
        
        causal_keywords = [
            ("sent", "received"),
            ("filed", "served"),
            ("requested", "provided"),
            ("scheduled", "attended")
        ]
        
        for cause_word, effect_word in causal_keywords:
            if cause_word in desc1 and effect_word in desc2:
                score += 0.2
        
        return min(score, 1.0)
    
    async def _analyze_timeline_gaps(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gaps and anomalies in timeline"""
        gap_analysis = {
            "significant_gaps": [],
            "missing_periods": [],
            "anomalies": []
        }
        
        if len(events) < 2:
            return gap_analysis
        
        sorted_events = sorted([e for e in events if e.get('date')], 
                             key=lambda x: x['date'])
        
        # Analyze gaps between events
        for i in range(1, len(sorted_events)):
            prev_date = datetime.strptime(sorted_events[i-1]['date'], '%Y-%m-%d')
            curr_date = datetime.strptime(sorted_events[i]['date'], '%Y-%m-%d')
            gap_days = (curr_date - prev_date).days
            
            if gap_days > 60:  # Significant gap
                gap_analysis["significant_gaps"].append({
                    "start_date": sorted_events[i-1]['date'],
                    "end_date": sorted_events[i]['date'],
                    "gap_days": gap_days,
                    "gap_significance": min(gap_days / 365, 1.0),  # Normalize to year
                    "before_event": sorted_events[i-1]['event_description'],
                    "after_event": sorted_events[i]['event_description']
                })
        
        # Identify potential missing periods based on event patterns
        if len(sorted_events) >= 5:
            # Calculate average gap
            gaps = []
            for i in range(1, len(sorted_events)):
                prev_date = datetime.strptime(sorted_events[i-1]['date'], '%Y-%m-%d')
                curr_date = datetime.strptime(sorted_events[i]['date'], '%Y-%m-%d')
                gaps.append((curr_date - prev_date).days)
            
            avg_gap = sum(gaps) / len(gaps)
            
            # Find unusually large gaps
            for i, gap in enumerate(gaps):
                if gap > avg_gap * 3:  # 3x average gap
                    gap_analysis["anomalies"].append({
                        "type": "unusually_large_gap",
                        "gap_days": gap,
                        "average_gap": avg_gap,
                        "ratio": gap / avg_gap,
                        "position": i + 1
                    })
        
        return gap_analysis
    
    async def _ai_timeline_analysis(self, content: str, events: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for advanced timeline analysis"""
        if not self.ai_service:
            return {}
        
        try:
            # Prepare events summary for AI
            events_summary = []
            for event in events[:10]:  # Limit to prevent token overflow
                events_summary.append({
                    "date": event.get('date'),
                    "type": event.get('event_type'),
                    "description": event.get('event_description', '')[:100]
                })
            
            prompt = f"""
Analyze this timeline of events for legal significance:

Events: {json.dumps(events_summary, indent=2)}
Document Content: {content[:1500]}

Provide timeline analysis in JSON format:
{{
    "timeline_narrative": "coherent story connecting the events",
    "key_turning_points": ["event1", "event2"],
    "causal_chains": [["cause", "effect", "result"]],
    "timeline_strength": "strong|moderate|weak",
    "missing_evidence_periods": ["period1", "period2"],
    "chronological_inconsistencies": ["issue1", "issue2"],
    "strategic_timeline_use": "how to use this timeline strategically",
    "corroborating_evidence_needed": ["what", "would", "strengthen"],
    "timeline_vulnerabilities": ["weakness1", "weakness2"]
}}
"""
            
            response = await self.ai_service.provider.generate_completion(
                prompt,
                "You are an expert legal timeline analyst and trial strategist."
            )
            
            if response.success:
                try:
                    return json.loads(response.content)
                except json.JSONDecodeError:
                    return {"ai_summary": response.content}
            
        except Exception as e:
            self.logger.error(f"AI timeline analysis failed: {e}")
        
        return {}
    
    def calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence in timeline analysis"""
        confidence = 0.4  # Base confidence
        
        events = analysis_data.get("temporal_events", [])
        if events:
            # Confidence increases with number of events
            event_confidence = min(len(events) * 0.1, 0.3)
            confidence += event_confidence
            
            # Confidence increases with date confidence
            avg_date_confidence = sum(e.get('date_confidence', 0.5) for e in events) / len(events)
            confidence += avg_date_confidence * 0.2
        
        patterns = analysis_data.get("pattern_analysis", {}).get("patterns", [])
        if patterns:
            confidence += min(len(patterns) * 0.05, 0.2)
        
        ai_analysis = analysis_data.get("ai_analysis", {})
        if ai_analysis:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_temporal_strength(self, findings: Dict[str, Any]) -> float:
        """Calculate temporal evidence strength"""
        strength = 0.3  # Base strength
        
        events = findings.get("temporal_events", [])
        if events:
            # Strength increases with event significance
            avg_significance = sum(e.get('significance_score', 0.5) for e in events) / len(events)
            strength += avg_significance * 0.4
            
            # Strength increases with chronological coherence
            chronological = findings.get("chronological_sequence", [])
            if len(chronological) > 1:
                strength += 0.2
        
        patterns = findings.get("pattern_analysis", {}).get("patterns", [])
        pattern_strength = sum(p.get('significance', 0.5) for p in patterns)
        strength += min(pattern_strength * 0.1, 0.3)
        
        ai_analysis = findings.get("ai_analysis", {})
        timeline_strength = ai_analysis.get("timeline_strength", "weak")
        if timeline_strength == "strong":
            strength += 0.2
        elif timeline_strength == "moderate":
            strength += 0.1
        
        return min(strength, 1.0)
    
    def extract_legal_significance(self, findings: Dict[str, Any]) -> str:
        """Extract legal significance from timeline analysis"""
        events = findings.get("temporal_events", [])
        patterns = findings.get("pattern_analysis", {}).get("patterns", [])
        ai_analysis = findings.get("ai_analysis", {})
        
        if not events:
            return "No temporal events identified"
        
        event_count = len(events)
        pattern_count = len(patterns)
        
        timeline_strength = ai_analysis.get("timeline_strength", "unknown")
        
        significance = f"Timeline contains {event_count} temporal events"
        
        if pattern_count > 0:
            significance += f" with {pattern_count} identified patterns"
        
        if timeline_strength != "unknown":
            significance += f" showing {timeline_strength} chronological coherence"
        
        return significance + "."
    
    def _generate_recommendations(self, findings: Dict[str, Any]) -> List[str]:
        """Generate timeline-based recommendations"""
        recommendations = []
        
        events = findings.get("temporal_events", [])
        gap_analysis = findings.get("gap_analysis", {})
        ai_analysis = findings.get("ai_analysis", {})
        
        # Event-based recommendations
        if len(events) < 3:
            recommendations.append("Seek additional temporal evidence to strengthen timeline")
        
        # Gap-based recommendations
        significant_gaps = gap_analysis.get("significant_gaps", [])
        if significant_gaps:
            recommendations.append(f"Investigate {len(significant_gaps)} significant timeline gaps")
        
        # AI recommendations
        missing_periods = ai_analysis.get("missing_evidence_periods", [])
        if missing_periods:
            recommendations.append(f"Focus discovery on missing periods: {', '.join(missing_periods[:2])}")
        
        corroborating_needed = ai_analysis.get("corroborating_evidence_needed", [])
        if corroborating_needed:
            recommendations.append(f"Seek corroborating evidence: {', '.join(corroborating_needed[:2])}")
        
        strategic_use = ai_analysis.get("strategic_timeline_use", "")
        if strategic_use:
            recommendations.append(f"Strategic approach: {strategic_use}")
        
        return recommendations
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> AgentResult:
        """Create an error result"""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResult(
            agent_name=self.name,
            analysis_type="timeline_analysis",
            confidence=0.0,
            findings={"error": error_message},
            recommendations=["Review input data and try again"],
            evidence_strength=0.0,
            legal_significance="Analysis failed",
            processing_time=processing_time,
            timestamp=datetime.now(),
            metadata={"error": True}
        )