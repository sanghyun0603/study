# Generated by Django 2.1.3 on 2019-05-09 08:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Movie',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=200)),
                ('year', models.IntegerField()),
                ('img', models.ImageField(blank=True, null=True, upload_to='')),
                ('text', models.CharField(max_length=1000)),
            ],
        ),
        migrations.CreateModel(
            name='Recomm',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.FloatField()),
                ('movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movieRec.Movie')),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Viewed',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating', models.FloatField()),
                ('movie', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movieRec.Movie')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movieRec.User')),
            ],
        ),
        migrations.AddField(
            model_name='recomm',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='movieRec.User'),
        ),
    ]