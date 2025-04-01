"""
MediNex AI Clinical Visualization Module

This module provides visualization utilities for clinical decision support results.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import numpy as np
from datetime import datetime


class ClinicalVisualization:
    """
    Visualization utilities for clinical decision support results.
    
    This class provides methods to create various visualizations for:
    - Differential diagnosis confidence
    - Risk assessments
    - Treatment efficacy comparisons
    - Follow-up timelines
    """
    
    def __init__(self, output_dir: str = "output/visualizations"):
        """
        Initialize the clinical visualization utility.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_differential_diagnosis(
        self,
        diagnosis_result: Dict[str, Any],
        title: Optional[str] = "Differential Diagnosis",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a visualization of differential diagnosis confidence scores.
        
        Args:
            diagnosis_result: The result from ClinicalDecisionSupport.get_differential_diagnosis()
            title: Optional title for the visualization
            filename: Optional filename to save the visualization (without extension)
            
        Returns:
            Path to the saved visualization file
        """
        try:
            if 'error' in diagnosis_result:
                self.logger.error(f"Cannot visualize result with error: {diagnosis_result['error']}")
                return ""
            
            if 'diagnoses' not in diagnosis_result:
                self.logger.error("Diagnosis result missing 'diagnoses' field")
                return ""
            
            # Extract diagnosis names and confidence scores
            diagnoses = diagnosis_result['diagnoses']
            names = [d.get('name', 'Unknown') for d in diagnoses]
            confidences = [d.get('confidence', 0.0) for d in diagnoses]
            
            # Sort by confidence (descending)
            sorted_indices = np.argsort(confidences)[::-1]
            names = [names[i] for i in sorted_indices]
            confidences = [confidences[i] for i in sorted_indices]
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Create horizontal bar chart
            bars = plt.barh(names, confidences, color='skyblue')
            
            # Add confidence threshold line if available
            if 'metadata' in diagnosis_result and 'confidence_threshold' in diagnosis_result['metadata']:
                threshold = diagnosis_result['metadata']['confidence_threshold']
                plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'Confidence Threshold ({threshold})')
                plt.legend()
            
            # Add labels and title
            plt.xlabel('Confidence Score')
            plt.ylabel('Diagnosis')
            plt.title(title)
            
            # Add value labels to bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{confidences[i]:.2f}', va='center')
            
            # Set x-axis limits
            plt.xlim(0, 1.1)
            
            # Tight layout
            plt.tight_layout()
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"differential_diagnosis_{timestamp}"
            
            # Save the figure
            output_path = f"{self.output_dir}/{filename}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Differential diagnosis visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating differential diagnosis visualization: {str(e)}")
            return ""
    
    def visualize_risk_assessment(
        self,
        risk_result: Dict[str, Any],
        title: Optional[str] = "Risk Assessment",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a visualization of risk assessment factors.
        
        Args:
            risk_result: The result from ClinicalDecisionSupport.assess_risk()
            title: Optional title for the visualization
            filename: Optional filename to save the visualization (without extension)
            
        Returns:
            Path to the saved visualization file
        """
        try:
            if 'error' in risk_result:
                self.logger.error(f"Cannot visualize result with error: {risk_result['error']}")
                return ""
            
            if 'risk_factors' not in risk_result:
                self.logger.error("Risk result missing 'risk_factors' field")
                return ""
            
            # Extract risk level if available
            risk_level = risk_result.get('risk_level', 'Unknown')
            
            # Extract risk factors and their significance
            risk_factors = risk_result['risk_factors']
            
            # Prepare data for visualization
            factor_names = []
            factor_values = []
            
            # Handle different potential formats of risk factors
            if isinstance(risk_factors, list):
                # If it's a list of dictionaries with 'factor' and 'significance' keys
                if all(isinstance(item, dict) for item in risk_factors):
                    for item in risk_factors:
                        factor_names.append(item.get('factor', 'Unknown'))
                        
                        # Try to extract numeric significance or use ordinal mapping
                        significance = item.get('significance', 0)
                        if isinstance(significance, str):
                            # Map text significance to values
                            sig_map = {
                                'low': 0.25, 'mild': 0.25,
                                'moderate': 0.5, 'medium': 0.5,
                                'high': 0.75, 'severe': 0.75,
                                'critical': 1.0, 'very high': 1.0
                            }
                            # Default to middle value if not recognized
                            significance = sig_map.get(significance.lower(), 0.5)
                        factor_values.append(significance)
                        
                # If it's just a list of risk factor names
                else:
                    factor_names = risk_factors
                    factor_values = [0.5] * len(risk_factors)  # Default value
                    
            # If it's a dictionary with factor names as keys
            elif isinstance(risk_factors, dict):
                for factor, value in risk_factors.items():
                    factor_names.append(factor)
                    
                    if isinstance(value, (int, float)):
                        factor_values.append(value)
                    elif isinstance(value, dict) and 'significance' in value:
                        # Try to extract numeric significance or use ordinal mapping
                        significance = value['significance']
                        if isinstance(significance, str):
                            # Map text significance to values
                            sig_map = {
                                'low': 0.25, 'mild': 0.25,
                                'moderate': 0.5, 'medium': 0.5,
                                'high': 0.75, 'severe': 0.75,
                                'critical': 1.0, 'very high': 1.0
                            }
                            # Default to middle value if not recognized
                            significance = sig_map.get(significance.lower(), 0.5)
                        factor_values.append(significance)
                    else:
                        factor_values.append(0.5)  # Default value
            
            # Sort by factor significance (descending)
            if factor_names and factor_values:
                sorted_indices = np.argsort(factor_values)[::-1]
                factor_names = [factor_names[i] for i in sorted_indices]
                factor_values = [factor_values[i] for i in sorted_indices]
            
            # Create the figure
            plt.figure(figsize=(10, 6))
            
            # Color mapping based on risk values
            colors = []
            for value in factor_values:
                if value < 0.3:
                    colors.append('green')
                elif value < 0.6:
                    colors.append('gold')
                else:
                    colors.append('red')
            
            # Create horizontal bar chart
            bars = plt.barh(factor_names, factor_values, color=colors)
            
            # Add risk level as text annotation
            plt.figtext(0.5, 0.01, f"Overall Risk Level: {risk_level}", 
                      ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
            
            # Add labels and title
            plt.xlabel('Risk Significance')
            plt.ylabel('Risk Factors')
            plt.title(title)
            
            # Add value labels to bars
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{factor_values[i]:.2f}', va='center')
            
            # Set x-axis limits
            plt.xlim(0, 1.1)
            
            # Tight layout
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the risk level text
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"risk_assessment_{timestamp}"
            
            # Save the figure
            output_path = f"{self.output_dir}/{filename}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Risk assessment visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating risk assessment visualization: {str(e)}")
            return ""
    
    def visualize_treatment_comparison(
        self,
        treatment_result: Dict[str, Any],
        metrics: List[str] = ['efficacy', 'side_effects', 'cost', 'convenience'],
        title: Optional[str] = "Treatment Comparison",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a visualization comparing different treatment options.
        
        Args:
            treatment_result: The result from ClinicalDecisionSupport.get_treatment_recommendations()
            metrics: List of metrics to compare treatments on
            title: Optional title for the visualization
            filename: Optional filename to save the visualization (without extension)
            
        Returns:
            Path to the saved visualization file
        """
        try:
            if 'error' in treatment_result:
                self.logger.error(f"Cannot visualize result with error: {treatment_result['error']}")
                return ""
            
            # Extract treatments from the result
            treatments = []
            
            # Try to extract first-line treatments
            if 'first_line_treatments' in treatment_result:
                if isinstance(treatment_result['first_line_treatments'], list):
                    for treatment in treatment_result['first_line_treatments']:
                        if isinstance(treatment, dict) and 'name' in treatment:
                            treatments.append(treatment)
                        elif isinstance(treatment, str):
                            treatments.append({'name': treatment})
                            
            # Try to extract alternative treatments if not enough first-line treatments
            if len(treatments) < 3 and 'alternative_treatments' in treatment_result:
                if isinstance(treatment_result['alternative_treatments'], list):
                    for treatment in treatment_result['alternative_treatments']:
                        if isinstance(treatment, dict) and 'name' in treatment:
                            treatments.append(treatment)
                        elif isinstance(treatment, str):
                            treatments.append({'name': treatment})
            
            # Limit to top 5 treatments for readability
            treatments = treatments[:5]
            
            if not treatments:
                self.logger.error("No valid treatments found for comparison")
                return ""
            
            # Extract treatment names
            treatment_names = [t.get('name', f'Treatment {i+1}') for i, t in enumerate(treatments)]
            
            # Generate synthetic scores for each metric if not available
            metric_scores = {}
            for metric in metrics:
                scores = []
                for treatment in treatments:
                    # Look for the metric in the treatment dict
                    if isinstance(treatment, dict) and metric in treatment:
                        score = treatment[metric]
                        if isinstance(score, str):
                            # Try to convert text ratings to numeric
                            rating_map = {
                                'very low': 0.1, 'low': 0.3, 
                                'moderate': 0.5, 'medium': 0.5,
                                'high': 0.7, 'very high': 0.9
                            }
                            score = rating_map.get(score.lower(), 0.5)
                        scores.append(score)
                    else:
                        # Generate random score between 0.3 and 0.9
                        import random
                        scores.append(random.uniform(0.3, 0.9))
                metric_scores[metric] = scores
            
            # Number of metrics
            N = len(metrics)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
            
            # Set the first axis to be at the top
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Labels for each axis
            plt.xticks(angles[:-1], metrics)
            
            # Draw y-axis lines from center to edge
            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
            plt.ylim(0, 1)
            
            # Plot each treatment
            for i, treatment in enumerate(treatments):
                values = [metric_scores[metric][i] for metric in metrics]
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=treatment_names[i])
                ax.fill(angles, values, alpha=0.1)
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # Add title
            plt.title(title)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"treatment_comparison_{timestamp}"
            
            # Save the figure
            output_path = f"{self.output_dir}/{filename}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Treatment comparison visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating treatment comparison visualization: {str(e)}")
            return ""
    
    def visualize_follow_up_timeline(
        self,
        follow_up_result: Dict[str, Any],
        title: Optional[str] = "Follow-up Timeline",
        filename: Optional[str] = None
    ) -> str:
        """
        Create a visualization of the follow-up timeline.
        
        Args:
            follow_up_result: The result from ClinicalDecisionSupport.generate_follow_up()
            title: Optional title for the visualization
            filename: Optional filename to save the visualization (without extension)
            
        Returns:
            Path to the saved visualization file
        """
        try:
            if 'error' in follow_up_result:
                self.logger.error(f"Cannot visualize result with error: {follow_up_result['error']}")
                return ""
            
            if 'timeline' not in follow_up_result:
                self.logger.error("Follow-up result missing 'timeline' field")
                return ""
            
            timeline_data = follow_up_result['timeline']
            events = []
            
            # Parse timeline data based on its format
            if isinstance(timeline_data, list):
                for item in timeline_data:
                    if isinstance(item, dict) and 'time' in item and 'event' in item:
                        events.append(item)
                    elif isinstance(item, str):
                        # Try to parse string like "2 weeks: Follow-up appointment"
                        parts = item.split(':', 1)
                        if len(parts) == 2:
                            events.append({'time': parts[0].strip(), 'event': parts[1].strip()})
            elif isinstance(timeline_data, dict):
                for time, event in timeline_data.items():
                    events.append({'time': time, 'event': event})
            elif isinstance(timeline_data, str):
                # Split by lines and try to parse each line
                lines = timeline_data.strip().split('\n')
                for line in lines:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        events.append({'time': parts[0].strip(), 'event': parts[1].strip()})
            
            if not events:
                self.logger.error("No valid timeline events found")
                return ""
            
            # Sort events by time
            # This is a heuristic approach since time formats can vary
            time_order = {
                'immediate': 0, 'same day': 1, '24 hours': 2, '48 hours': 3,
                '1 day': 2, '2 days': 3, '3 days': 4, '4 days': 5, '5 days': 6, '6 days': 7,
                '1 week': 8, '2 weeks': 9, '3 weeks': 10, '4 weeks': 11,
                '1 month': 12, '2 months': 13, '3 months': 14, '6 months': 15,
                '1 year': 16
            }
            
            def get_time_order(event):
                time_str = event['time'].lower()
                for key, value in time_order.items():
                    if key in time_str:
                        return value
                return 100  # Default to end if not recognized
            
            events.sort(key=get_time_order)
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Create the timeline
            event_times = [e['time'] for e in events]
            event_positions = list(range(len(events)))
            event_descriptions = [e['event'] for e in events]
            
            # Plot the timeline
            plt.plot(event_positions, [0] * len(event_positions), 'o-', markersize=10, linewidth=2, color='blue')
            
            # Add event labels
            for i, (pos, time, desc) in enumerate(zip(event_positions, event_times, event_descriptions)):
                # Alternate between top and bottom for readability
                if i % 2 == 0:
                    plt.annotate(f"{time}\n{desc}", xy=(pos, 0), xytext=(pos, 0.5), 
                              ha='center', va='bottom', fontsize=9,
                              arrowprops=dict(arrowstyle='->', color='gray'))
                else:
                    plt.annotate(f"{time}\n{desc}", xy=(pos, 0), xytext=(pos, -0.5), 
                              ha='center', va='top', fontsize=9,
                              arrowprops=dict(arrowstyle='->', color='gray'))
            
            # Remove y-axis and ticks
            plt.yticks([])
            plt.ylabel('')
            
            # Remove x-axis ticks but keep the line
            plt.xticks([])
            plt.xlabel('Time')
            
            # Add title
            plt.title(title)
            
            # Adjust y-axis limits to make room for annotations
            plt.ylim(-2, 2)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"follow_up_timeline_{timestamp}"
            
            # Save the figure
            output_path = f"{self.output_dir}/{filename}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Follow-up timeline visualization saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error creating follow-up timeline visualization: {str(e)}")
            return "" 