import { Button } from "@/components/ui/button";
import { TrendingUp, PieChart, DollarSign, BarChart3 } from "lucide-react";

interface SuggestionChipsProps {
  onSuggestionClick: (suggestion: string) => void;
  disabled?: boolean;
}

const suggestions = [
  { icon: TrendingUp, text: "Analyze Infosys stock" },
  { icon: PieChart, text: "Suggest a balanced portfolio" },
  { icon: DollarSign, text: "Compare tech sector stocks" },
  { icon: BarChart3, text: "Market trends this quarter" },
];

export const SuggestionChips = ({
  onSuggestionClick,
  disabled,
}: SuggestionChipsProps) => {
  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {suggestions.map((suggestion, index) => {
        const Icon = suggestion.icon;
        return (
          <Button
            key={index}
            variant="outline"
            size="sm"
            onClick={() => onSuggestionClick(suggestion.text)}
            disabled={disabled}
            className="bg-card/50 border-border/50 hover:bg-card hover:border-primary/50 hover:text-primary transition-all backdrop-blur-sm"
          >
            <Icon className="w-4 h-4 mr-2" />
            {suggestion.text}
          </Button>
        );
      })}
    </div>
  );
};
