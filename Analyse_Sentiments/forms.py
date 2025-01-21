from django import forms


class SentimentForm(forms.Form):
    text = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'class': (
                    'w-full px-4 py-3 border border-gray-300 rounded-lg '
                    'focus:ring-2 focus:ring-pink-300 focus:outline-none '
                    'focus:bg-gradient-to-r from-pink-100 via-purple-50 to-purple-200 '
                    'transition-shadow duration-300 shadow-md hover:shadow-lg'
                ),
                'placeholder': 'Write a text here...',
                'rows': 5  # Définit la hauteur du champ texte
            }
        ),
        label=""
    )
